"""Fonctions de tâche (observations, récompenses, terminaisons) propres à l'escalade.

Chaque fonction reçoit `env` et rend un tenseur (num_envs, ...) : elles calculent pour tous les robots à la fois.
"Accroché" a le même sens que `ClimbingSim.hooked()` du simulateur : centre de la boule à moins de 6 mm du centre
de la coupe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from climbing_rl.robot import BALL_SITES, hang_pose
from climbing_rl.wall import LIMBS, Wall

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

HOOK_TOL = 0.006
HOLD_STEPS = 5  # pas de politique (5 x 20 ms = 0,1 s) où une boule doit rester dans une prise pour valider un record
_ROBOT = SceneEntityCfg("robot")


def _robot(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["robot"]


def _ball_site_ids(env: ManagerBasedRlEnv) -> list[int]:
  """Indices (dans l'entité robot) des sites des 4 boules, ordre LIMBS. Calculés une fois puis gardés."""
  ids = getattr(env, "_climb_ball_site_ids", None)
  if ids is None:
    ids, _ = _robot(env).find_sites(BALL_SITES, preserve_order=True)
    env._climb_ball_site_ids = ids
  return ids


def _targets(env: ManagerBasedRlEnv, sockets: Sequence[Sequence[float]]) -> torch.Tensor:
  """Les positions des coupes visées (4, 3) en tenseur, sur le bon appareil."""
  return torch.tensor(sockets, dtype=torch.float32, device=env.device)


def ball_positions(env: ManagerBasedRlEnv) -> torch.Tensor:
  """(N, 4, 3) positions monde des 4 boules."""
  return _robot(env).data.site_pos_w[:, _ball_site_ids(env)]


def ball_distances(env: ManagerBasedRlEnv, sockets) -> torch.Tensor:
  """(N, 4) distance de chaque boule à sa coupe cible."""
  return torch.norm(ball_positions(env) - _targets(env, sockets), dim=-1)


# ==== Tâches à cibles fixes (Hang, LeftFoot, BothFeet) =====================================================

# ---- observations (pour le critic : vérité terrain, inconnue du vrai robot) --------------------------------
def ball_offsets(env: ManagerBasedRlEnv, sockets) -> torch.Tensor:
  """(N, 12) vecteur boule -> coupe cible pour chaque membre."""
  return (_targets(env, sockets) - ball_positions(env)).reshape(env.num_envs, -1)


def hooked_flags(env: ManagerBasedRlEnv, sockets, tol: float = HOOK_TOL) -> torch.Tensor:
  """(N, 4) 1 si la boule est dans sa coupe cible, 0 sinon."""
  return (ball_distances(env, sockets) < tol).float()


# ---- récompenses -----------------------------------------------------------------------------------------
def balls_near_sockets(env: ManagerBasedRlEnv, sockets, std: float = 0.01, limbs: Sequence[int] = (0, 1, 2, 3)):
  """Moyenne sur les membres choisis de exp(-d² / std²) : 1 quand la boule est au centre de sa coupe."""
  d = ball_distances(env, sockets)[:, list(limbs)]
  return torch.exp(-torch.square(d / std)).mean(dim=-1)


def balls_hooked(env: ManagerBasedRlEnv, sockets, tol: float = HOOK_TOL, limbs: Sequence[int] = (0, 1, 2, 3)):
  """Fraction des membres choisis accrochés à leur coupe cible."""
  return (ball_distances(env, sockets)[:, list(limbs)] < tol).float().mean(dim=-1)


def limb_reach(env: ManagerBasedRlEnv, sockets, limb: int, std: float = 0.03):
  """exp(-d² / std²) pour UN membre vers sa cible : guide le mouvement de loin (std plus grand)."""
  d = ball_distances(env, sockets)[:, limb]
  return torch.exp(-torch.square(d / std))


def limb_hooked(env: ManagerBasedRlEnv, sockets, limb: int, tol: float = HOOK_TOL):
  """1 si ce membre est accroché à sa cible."""
  return (ball_distances(env, sockets)[:, limb] < tol).float()


# ---- terminaisons ----------------------------------------------------------------------------------------
def trunk_dropped(env: ManagerBasedRlEnv, start_z: float, max_drop: float = 0.05) -> torch.Tensor:
  """Le tronc est descendu de plus de max_drop sous sa hauteur de départ : le robot a lâché."""
  return _robot(env).data.root_link_pos_w[:, 2] < start_z - max_drop


def limbs_lost(env: ManagerBasedRlEnv, sockets, limbs: Sequence[int], max_dist: float = 0.03) -> torch.Tensor:
  """Un des membres qui doivent rester accrochés s'est éloigné de plus de max_dist de sa coupe."""
  return (ball_distances(env, sockets)[:, list(limbs)] > max_dist).any(dim=-1)


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

  # Ce que lisent les récompenses et terminaisons
  n_hooked = hooked.sum(dim=-1)
  env._climb_hooked = hooked                                            # (N, 4) bool
  env._climb_n_hooked = n_hooked                                        # (N,)   nombre de boules accrochées
  env._climb_others_hooked = (n_hooked[:, None] - hooked.long()) >= 3   # (N, 4) les 3 AUTRES membres sont accrochés
  env._climb_new_record = new_record                                    # (N, 4) bool, vrai au pas où le membre bat son record
  env._climb_cand_dist = cand_dist                                      # (N, 4) distance à la candidate la plus proche (inf si aucune)
  env._climb_trunk_z = z                                                # (N,)   hauteur actuelle du tronc
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
  return env._climb_n_hooked < 2

def trunk_height_record(env: ManagerBasedRlEnv) -> torch.Tensor:
  # Reward : Gain de record de hauteur du tronc à ce pas
  _update_state(env)
  return env._climb_trunk_gain

def next_hold_rung(env: ManagerBasedRlEnv, std: float = 0.04) -> torch.Tensor:
  # Reward : On calcul la gaussienne de la distance d'une prise à sa candidate, comptée seulement si les 3 autres membres tiennent (maximum sur les 4 membres)
  _update_state(env)
  # On fait torch.exp(-torch.square(env._climb_cand_dist/std)) pour calculer la gaussienne de la distance de chaque boule vers la prise cible
  # On multiplie par env._climb_others_hooked.float() pour mettre à 0 les membres dont les 3 autre ne sont pas attachés
  # On garde uniquement le plus grand des 4 avec .max(dim=-1).values
  return (torch.exp(-torch.square(env._climb_cand_dist/std)) * env._climb_others_hooked.float()).max(dim=-1).values

def member_hold_record(env: ManagerBasedRlEnv) -> torch.Tensor:
  # Reward : Nombre de membres qui battent leur record à ce pas
  _update_state(env)
  # On renvoi le nombre de membre de ce robot qui batte leur record en faisant la somme torch.sum du tenseur _climb_new_record
  return torch.sum(env._climb_new_record, dim=-1).float()
