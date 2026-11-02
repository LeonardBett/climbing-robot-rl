from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from climbing_rl.robot import hang_pose
from climbing_rl.tasks.mdp import HOLD_STEPS, HOOK_TOL, LOST_TOL, _robot, ball_positions
from climbing_rl.wall import LIMBS, Hold, Wall

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ==== Tâche ClimbN1 : état d'escalade, rangé sur env =======================================================
# Règle : les récompenses et terminaisons lisent ce que _update_state a calculé (une fois par pas, avant les
# resets). Les observations recalculent à partir de l'état actuel, car elles passent APRÈS les resets.

def _ensure_state(env: ManagerBasedRlEnv, wall_name: str) -> None:
  """Crée au premier appel les données du mur et les tenseurs d'état (une ligne par robot)."""
  if hasattr(env, "_climb_sockets"):
    return
  wall = Wall.load(wall_name)
  holds = list(wall.holds)
  # Données du mur, identiques pour tous les robots
  env._climb_sockets = torch.tensor([h.socket.tolist() for h in holds], dtype=torch.float32, device=env.device)  # (H, 3)
  env._climb_hold_rows = torch.tensor([h.row for h in holds], dtype=torch.long, device=env.device)  # (H,)
  env._climb_start_rows = torch.tensor([wall.start[l].row for l in LIMBS], dtype=torch.long, device=env.device)  # (4,)
  env._climb_start_z = float(hang_pose(wall_name).trunk_pos[2])
  # État par robot
  n = env.num_envs
  env._climb_limb_record = env._climb_start_rows.repeat(n, 1)                         # (N, 4) rangée de la plus haute prise validée
  env._climb_trunk_record = torch.full((n,), env._climb_start_z, device=env.device)   # (N,) hauteur max du tronc
  env._climb_hook_count = torch.zeros(n, 4, dtype=torch.long, device=env.device)      # (N, 4) pas d'affilée dans une prise du barreau suivant
  env._climb_last_step = -1                                                            # dernier pas traité par _update_state

  env._climb_approach_record = torch.full((n, 4), float("inf"), device=env.device)  # (N, 4) plus petite distance atteinte vers une candidate


def _check_state(env: ManagerBasedRlEnv) -> None:
  if not hasattr(env, "_climb_sockets"):
    raise RuntimeError("état d'escalade absent : ajouter l'event reset_record (mode='reset') dans cfg.events")


# ---- event -----------------------------------------------------------------------------------------------
def reset_record(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, wall_name: str) -> None:
  """Event de reset : remet les records des robots env_ids à leur état de départ."""
  _ensure_state(env, wall_name)
  if env_ids is None:  # certains appels passent None pour dire « tous »
    env_ids = slice(None)
  env._climb_limb_record[env_ids] = env._climb_start_rows
  env._climb_trunk_record[env_ids] = env._climb_start_z
  env._climb_hook_count[env_ids] = 0
  env._climb_approach_record[env_ids] = float("inf")


# ---- calculs géométriques sur l'état actuel ---------------------------------------------------------------
def _nearest(env: ManagerBasedRlEnv):
  """Boules (N,4,3), distances à toutes les coupes (N,4,H), distance (N,4) et indice (N,4) de la plus proche."""
  balls = ball_positions(env)
  d = torch.norm(env._climb_sockets[None, None] - balls[:, :, None], dim=-1)
  dist, idx = d.min(dim=-1)
  return balls, d, dist, idx


def _next_rows(env: ManagerBasedRlEnv) -> torch.Tensor:
  """(N, 4) rangée du barreau juste au-dessus du record de chaque membre (très grand nombre s'il n'y en a plus)."""
  rows = env._climb_hold_rows[None, None, :]                       # (1, 1, H)
  above = rows > env._climb_limb_record[:, :, None]                 # (N, 4, H)
  big = torch.iinfo(torch.long).max
  return torch.where(above, rows, torch.full_like(rows, big)).min(dim=-1).values


def _candidates(env: ManagerBasedRlEnv, d: torch.Tensor):
  """Distance (N, 4) à la prise candidate la plus proche (inf s'il n'y en a pas) et son indice (N, 4)."""
  is_cand = env._climb_hold_rows[None, None, :] == _next_rows(env)[:, :, None]  # (N, 4, H)
  d_cand = torch.where(is_cand, d, torch.full_like(d, float("inf")))
  return d_cand.min(dim=-1)


# ---- mise à jour, une seule fois par pas -------------------------------------------------------------------
def _update_state(env: ManagerBasedRlEnv) -> None:
  """Met à jour records et compteurs, et range sur env ce que lisent les récompenses et terminaisons."""
  _check_state(env)
  step = env.common_step_counter
  if step == env._climb_last_step:
    return
  env._climb_last_step = step

  balls, d, dist, idx = _nearest(env)
  hooked = dist < HOOK_TOL                          # (N, 4) boule dans une coupe, n'importe laquelle
  hooked_rows = env._climb_hold_rows[idx]           # (N, 4) rangée de cette coupe

  # Record d'un membre : accroché à une prise du barreau suivant pendant HOLD_STEPS pas d'affilée
  in_next = hooked & (hooked_rows == _next_rows(env))
  zeros = torch.zeros_like(env._climb_hook_count)
  count = torch.where(in_next, env._climb_hook_count + 1, zeros)
  new_record = count >= HOLD_STEPS
  env._climb_limb_record = torch.where(new_record, hooked_rows, env._climb_limb_record)
  env._climb_hook_count = torch.where(new_record, zeros, count)

  # Record du tronc
  z = _robot(env).data.root_link_pos_w[:, 2]
  env._climb_trunk_gain = torch.clamp(z - env._climb_trunk_record, min=0.0)
  env._climb_trunk_record = torch.maximum(env._climb_trunk_record, z)

  # Candidates, APRÈS la mise à jour des records : un membre qui vient de valider vise déjà le barreau suivant
  cand_dist, _ = _candidates(env, d)

  # Record d'approche : plus petite distance atteinte vers une candidate du barreau suivant.
  # Remis à inf quand le membre valide une prise (ses candidates changent de barreau).
  prev = torch.where(new_record, torch.full_like(cand_dist, float("inf")), env._climb_approach_record)
  # Gain = de combien le record s'est amélioré à ce pas. Au premier pas (prev = inf), on initialise sans récompenser.
  env._climb_approach_gain = torch.where(
    torch.isfinite(prev), torch.clamp(prev - cand_dist, min=0.0), torch.zeros_like(cand_dist)
  )  # (N, 4) en mètres
  env._climb_approach_record = torch.minimum(prev, cand_dist)

  # Ce que lisent les récompenses et terminaisons
  n_hooked = hooked.sum(dim=-1)
  env._climb_hooked = hooked                                            # (N, 4) bool
  env._climb_n_hooked = n_hooked                                        # (N,)   nombre de boules accrochées
  env._climb_others_hooked = (n_hooked[:, None] - hooked.long()) >= 3   # (N, 4) les 3 AUTRES membres sont accrochés
  env._climb_new_record = new_record                                    # (N, 4) bool, vrai au pas où le membre bat son record
  env._climb_cand_dist = cand_dist                                      # (N, 4) distance à la candidate la plus proche (inf si aucune)
  env._climb_trunk_z = z                                                # (N,)   hauteur actuelle du tronc
  env._climb_n_held = (dist < LOST_TOL).sum(dim=-1)                     # (N,)   nombre de boules à moins de 3 cm d'une coupe
  # env._climb_trunk_gain (N,) et env._climb_trunk_record (N,) sont rangés plus haut


# ---- observations du critic (recalculées sur l'état actuel) ---------------------------------------------
def nearest_socket_offsets(env: ManagerBasedRlEnv, wall_name: str) -> torch.Tensor:
  """(N, 12) vecteur boule -> coupe la plus proche, pour chaque membre."""
  _ensure_state(env, wall_name)
  balls, _, _, idx = _nearest(env)
  return (env._climb_sockets[idx] - balls).reshape(env.num_envs, -1)


def hooked_any(env: ManagerBasedRlEnv, wall_name: str) -> torch.Tensor:
  """(N, 4) 1 si la boule est dans une coupe, quelle qu'elle soit."""
  _ensure_state(env, wall_name)
  _, _, dist, _ = _nearest(env)
  return (dist < HOOK_TOL).float()


def next_candidate_offsets(env: ManagerBasedRlEnv, wall_name: str) -> torch.Tensor:
  """(N, 12) vecteur boule -> prise candidate la plus proche du barreau suivant (0 s'il n'y en a plus)."""
  _ensure_state(env, wall_name)
  balls, d, _, _ = _nearest(env)
  cand_dist, cand_idx = _candidates(env, d)
  offset = env._climb_sockets[cand_idx] - balls
  offset = torch.where(torch.isfinite(cand_dist)[..., None], offset, torch.zeros_like(offset))
  return offset.reshape(env.num_envs, -1)

def terminated_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 au pas où l'épisode se termine par un échec (pas par time-out), 0 sinon. À utiliser avec un poids négatif."""
  return env.termination_manager.terminated.float()

# Travail personel : terminaisons et récompenses de la tache climb_N1, fait avec l'aide des exemples plus haut
# Chacune commence par _update_state(env) pour mettre à jour env avec les dernière valeur de la simulation (fait une seul fois par pas)
# Elle renvoi un tenseur (N,), un booléen pour une terminaison ou un float pour une récompense

def trunk_dropped_record(env: ManagerBasedRlEnv, max_drop: float = 0.05) -> torch.Tensor:
  # Terminaison : le tronc est descendu de plus de max_drop sous son record
  _update_state(env)
  return env._climb_trunk_z < env._climb_trunk_record - max_drop

def less_two_hooked(env: ManagerBasedRlEnv) -> torch.Tensor:
  # Terminaison : moins de 2 boules accrochés à une prise
  _update_state(env)
  return env._climb_n_held < 2

def trunk_height_record(env: ManagerBasedRlEnv) -> torch.Tensor:
  # Reward : Gain de record de hauteur du tronc à ce pas
  _update_state(env)
  return env._climb_trunk_gain

def next_hold_rung(env: ManagerBasedRlEnv, std: float = 0.04) -> torch.Tensor:
  # Reward : On recompense un membre si il s'approche de sa prise idéal et que les 3 autres reste attaché
  _update_state(env)
  # On recupere pour chaque membre du robot sont gain de distance avec la prise cible env._climb_approach_gain
  # On multiplie par env._climb_others_hooked.float() pour mettre à 0 les membres dont les 3 autre ne sont pas attachés
  # On garde uniquement le plus grand des 4 avec .max(dim=-1).values
  return (env._climb_approach_gain * env._climb_others_hooked.float()).max(dim=-1).values

def member_hold_record(env: ManagerBasedRlEnv) -> torch.Tensor:
  # Reward : Nombre de membres qui battent leur record à ce pas
  _update_state(env)
  # On renvoi le nombre de membre de ce robot qui batte leur record en faisant la somme torch.sum du tenseur _climb_new_record
  return torch.sum(env._climb_new_record, dim=-1).float()