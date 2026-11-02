from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from climbing_rl.robot import hang_pose
from climbing_rl.tasks.mdp import HOLD_STEPS, HOOK_TOL, _robot, ball_positions
from climbing_rl.wall import LIMBS, Hold, Wall

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ==== Tâche ClimbN1Route : route codée en dur =============================================================
# Motif de coups sur 2 barreaux (N1 se répète tous les 2 barreaux) : (membre, Δcolonne, Δrangée), relatif à la
# prise actuelle du membre. Pieds en zigzag, mains tout droit : À CONFIRMER (surtout les mains et le retour du pied
# gauche en +3). La route est déroulée depuis les prises de départ tant que les prises visées existent.
ROUTE_PATTERN = (
  ("left_foot", -3, 5), ("right_foot", 0, 5), ("right_hand", 0, 5), ("left_hand", 0, 5),
  ("left_foot", +3, 5), ("right_foot", 0, 5), ("right_hand", 0, 5), ("left_hand", 0, 5),
)


def _ensure_route(env: ManagerBasedRlEnv, wall_name: str) -> None:
  """Crée au premier appel la route (liste des coups) et les tenseurs d'état (une ligne par robot)."""
  if hasattr(env, "_route_sockets"):
    return
  wall = Wall.load(wall_name)
  holds = list(wall.holds)
  # Déroule le motif : pour chaque coup, quel membre et vers quelle prise (indice dans holds)
  current = {l: wall.start[l] for l in LIMBS}
  move_limb, move_hold = [], []
  finished = False
  while not finished:
    for limb, dcol, drow in ROUTE_PATTERN:
      target = Hold(current[limb].col + dcol, current[limb].row + drow)
      if target not in holds:
        finished = True
        break
      move_limb.append(LIMBS.index(limb))
      move_hold.append(holds.index(target))
      current[limb] = target
  if not move_limb:
    raise ValueError(f"route vide sur le mur {wall_name} : vérifier ROUTE_PATTERN")

  dev = env.device
  # Données du mur et de la route, identiques pour tous les robots
  env._route_sockets = torch.tensor([h.socket.tolist() for h in holds], dtype=torch.float32, device=dev)  # (H, 3)
  env._route_move_limb = torch.tensor(move_limb, dtype=torch.long, device=dev)   # (M,) membre de chaque coup
  env._route_move_hold = torch.tensor(move_hold, dtype=torch.long, device=dev)   # (M,) prise visée par chaque coup
  env._route_start_holds = torch.tensor([holds.index(wall.start[l]) for l in LIMBS], dtype=torch.long, device=dev)  # (4,)
  env._route_start_z = float(hang_pose(wall_name).trunk_pos[2])
  # État par robot
  n = env.num_envs
  env._route_move = torch.zeros(n, dtype=torch.long, device=dev)                  # (N,) indice du coup en cours
  env._route_limb_hold = env._route_start_holds.repeat(n, 1)                       # (N, 4) prise attribuée à chaque membre
  env._route_count = torch.zeros(n, dtype=torch.long, device=dev)                 # (N,) pas d'affilée du membre actif dans sa cible
  env._route_trunk_record = torch.full((n,), env._route_start_z, device=dev)      # (N,) hauteur max du tronc
  env._route_last_step = -1


def reset_route(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, wall_name: str) -> None:
  """Event de reset : premier coup de la route, prises de départ, record du tronc remis à zéro."""
  _ensure_route(env, wall_name)
  if env_ids is None:
    env_ids = slice(None)
  env._route_move[env_ids] = 0
  env._route_limb_hold[env_ids] = env._route_start_holds
  env._route_count[env_ids] = 0
  env._route_trunk_record[env_ids] = env._route_start_z


def _route_current(env: ManagerBasedRlEnv):
  """Coup en cours : membre actif (N,), position de sa cible (N, 3), masque actif (N, 4), route finie (N,)."""
  m = env._route_move_limb.numel()
  done = env._route_move >= m
  idx = env._route_move.clamp(max=m - 1)
  limb = env._route_move_limb[idx]
  target = env._route_sockets[env._route_move_hold[idx]]
  active = torch.nn.functional.one_hot(limb, 4).bool() & ~done[:, None]
  return limb, target, active, done


def _update_route(env: ManagerBasedRlEnv) -> None:
  """Une fois par pas : valide le coup en cours si besoin, puis range sur env ce que lisent récompenses et terminaisons."""
  if not hasattr(env, "_route_sockets"):
    raise RuntimeError("état de la route absent : ajouter l'event reset_route (mode='reset') dans cfg.events")
  step = env.common_step_counter
  if step == env._route_last_step:
    return
  env._route_last_step = step

  ar = torch.arange(env.num_envs, device=env.device)
  m = env._route_move_limb.numel()
  balls = ball_positions(env)  # (N, 4, 3)

  # Validation : la boule du membre actif reste HOLD_STEPS pas d'affilée dans sa cible
  limb, target, _, done = _route_current(env)
  d_target = torch.norm(balls[ar, limb] - target, dim=-1)
  in_target = (d_target < HOOK_TOL) & ~done
  count = torch.where(in_target, env._route_count + 1, torch.zeros_like(env._route_count))
  validated = count >= HOLD_STEPS

  # Coup validé : le membre est désormais attribué à sa cible, et on passe au coup suivant
  new_hold = env._route_move_hold[env._route_move.clamp(max=m - 1)]
  limb_hold = env._route_limb_hold.clone()
  limb_hold[ar, limb] = torch.where(validated, new_hold, limb_hold[ar, limb])
  env._route_limb_hold = limb_hold
  env._route_move = env._route_move + validated.long()
  env._route_count = torch.where(validated, torch.zeros_like(count), count)

  # Record du tronc
  z = _robot(env).data.root_link_pos_w[:, 2]
  env._route_trunk_gain = torch.clamp(z - env._route_trunk_record, min=0.0)
  env._route_trunk_record = torch.maximum(env._route_trunk_record, z)

  # État APRÈS le coup : nouveau membre actif et nouvelle cible
  limb, target, active, done = _route_current(env)
  d_new = torch.norm(balls[ar, limb] - target, dim=-1)
  env._route_active = active                                                                  # (N, 4) bool, membre dont c'est le tour
  env._route_d_target = torch.where(done, torch.full_like(d_new, float("inf")), d_new)       # (N,)   distance membre actif -> cible (inf si route finie)
  env._route_dist_assigned = torch.norm(balls - env._route_sockets[env._route_limb_hold], dim=-1)  # (N, 4) distance boule -> prise attribuée
  env._route_validated = validated                                                            # (N,)   bool, un coup validé à ce pas
  env._route_trunk_z = z                                                                      # (N,)


# ---- observations (recalculées sur l'état actuel) -------------------------------------------------------
def active_limb(env: ManagerBasedRlEnv, wall_name: str) -> torch.Tensor:
  """(N, 4) 1 pour le membre dont c'est le tour, ordre LIMBS. Connu aussi sur le vrai robot (c'est le plan)."""
  _ensure_route(env, wall_name)
  return _route_current(env)[2].float()


def cycling_ball_offsets(env: ManagerBasedRlEnv, wall_name: str) -> torch.Tensor:
  """(N, 12) vecteur boule -> prise attribuée, pour chaque membre."""
  _ensure_route(env, wall_name)
  return (env._route_sockets[env._route_limb_hold] - ball_positions(env)).reshape(env.num_envs, -1)


def active_limb_offset(env: ManagerBasedRlEnv, wall_name: str) -> torch.Tensor:
  """(N, 3) vecteur boule du membre actif -> sa cible (0 si la route est finie)."""
  _ensure_route(env, wall_name)
  limb, target, _, done = _route_current(env)
  ar = torch.arange(env.num_envs, device=env.device)
  off = target - ball_positions(env)[ar, limb]
  return torch.where(done[:, None], torch.zeros_like(off), off)


# Terminaisons et récompenses de la tache climb_N1, fait avec l'aide des exemples plus haut
# Chacune commence par _update_state(env) pour mettre à jour env avec les dernière valeur de la simulation (fait une seul fois par pas)
# Elle renvoi un tenseur (N,), un booléen pour une terminaison ou un float pour une récompense

def others_limbs_hooked(env: ManagerBasedRlEnv, std: float = 0.01) -> torch.Tensor:
  """Moyenne, sur les membres NON actifs, de exp(-(d/std)²) avec d = distance à leur prise attribuée."""
  _update_route(env)
  others = (~env._route_active).float()                                    # (N, 4) 1 pour les membres qui doivent tenir
  near = torch.exp(-torch.square(env._route_dist_assigned / std))          # (N, 4) 1 au centre de la coupe
  return (near * others).sum(dim=-1) / others.sum(dim=-1)                  # moyenne sur les 3 (ou 4 si route finie)


def active_limb_reach(env: ManagerBasedRlEnv, std: float = 0.04) -> torch.Tensor:
  """exp(-(d/std)²) avec d = distance du membre actif à sa cible (0 si la route est finie)."""
  _update_route(env)
  return torch.exp(-torch.square(env._route_d_target / std))


def active_limb_hooked(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 au pas où un coup est validé, 0 sinon (bonus ponctuel)."""
  _update_route(env)
  return env._route_validated.float()


def other_limbs_lost(env: ManagerBasedRlEnv, max_dist: float = 0.03) -> torch.Tensor:
  """Un des membres NON actifs est à plus de max_dist de sa prise attribuée."""
  _update_route(env)
  return ((env._route_dist_assigned > max_dist) & ~env._route_active).any(dim=-1)


def route_trunk_dropped(env: ManagerBasedRlEnv, max_drop: float = 0.05) -> torch.Tensor:
  """Le tronc est descendu de plus de max_drop sous son record."""
  _update_route(env)
  return env._route_trunk_z < env._route_trunk_record - max_drop