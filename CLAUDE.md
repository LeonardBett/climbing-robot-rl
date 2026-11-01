# Contexte pour Claude Code — climbing-robot-rl

Projet personnel de Léonard (Master ASPIC, Bordeaux), issu du hackathon Rock Climbing Robots : apprentissage par
renforcement du Microban grimpeur avec mjlab. Machine : Windows 11 + WSL Ubuntu, RTX 5070 (12 Go, sm_120).
L'utilisateur écrit en français : répondre en français. C'est sa première expérience en RL : il écrit lui-même ses
tâches ; expliquer ce qui est fait plutôt que tout réécrire à sa place.

## Règles de travail
- Ne jamais modifier `simulator/` : c'est un sous-module git (simulateur officiel, GPL-3.0) dont on lit seulement les
  fichiers de données.
- Avant tout changement important, vérifier que git est propre (`git status`).
- Après toute modification de la physique, du robot ou de la scène :
  `uv run check-scene --num-envs 64 --seconds 3` (option `--task <tâche>` pour une autre tâche).
  Attendu : 0 auto-contact au départ, aucun `nefc overflow`, les 4 membres accrochés à 100 %, « OK ».
- Un seul changement à la fois, puis test. Signaler l'effet sur les « Steps per second ».
- Ne pas lancer d'entraînement long sans le demander (il occupe le GPU de l'utilisateur).
- Rester ciblé : pas de mesures comparatives ou de benchmarks longs sauf demande explicite.

## Structure du dépôt
```
climbing-robot-rl/
├── src/climbing_rl/      le code
├── simulator/            sous-module : rock-climbing-robots/simulator
├── pyproject.toml        dépendances uv (BAM et microban_model en HTTPS)
└── README.md
```
`wall.py` trouve les données dans `<racine>/simulator/climbing_sim` (`Path(__file__).resolve().parents[2]`), ou via
la variable d'environnement `CLIMBING_SIM_DIR`.

## Architecture
mjlab 1.3 (RL massivement parallèle sur GPU, API type Isaac Lab, physique MuJoCo Warp), PPO via rsl_rl 5.0.1.

| Fichier | Rôle |
|---|---|
| `src/climbing_rl/wall.py` | lit les murs JSON et la géométrie des prises dans `simulator/climbing_sim/` ; formules des positions des coupes (recopiées de `simulator/climbing_sim/wall.py`) |
| `src/climbing_rl/robot.py` | robot = `microban_model/mjcf/robot_climb.xml` ; collisions "light" (boîtes) / "full" ; servos BAM ; pose accrochée (méthode de `ClimbingSim.reset_hanging` : `hang.json` + recalage SVD + IK) |
| `src/climbing_rl/scene.py` | ajoute sol, mur, prises (jusqu'à `max_row`) via `SceneCfg.spec_fn` |
| `src/climbing_rl/tasks/mdp.py` | fonctions d'escalade : distances boule/coupe, accroché (< 6 mm), récompenses, terminaisons |
| `src/climbing_rl/tasks/climb_env_cfg.py` | config des tâches (`_base_cfg` commun, `hang_env_cfg`, `left_foot_env_cfg`, `both_feet_env_cfg`) + `ppo_cfg` |
| `src/climbing_rl/tasks/__init__.py` | enregistre les tâches (point d'entrée `mjlab.tasks`) |
| `src/climbing_rl/check_scene.py` | test sans entraînement (`uv run check-scene`) |

Tâches : `Climb-Hang-Microban`, `Climb-LeftFoot-Microban`, `Climb-BothFeet-Microban`.

Commandes :
- `uv run train <tâche> --env.scene.num-envs 6144` (6144 tient dans 12 Go, ~33 000 pas/s sur BothFeet)
- reprise : `--agent.resume True --agent.load-run <dossier> --agent.load-checkpoint model_N.pt`
- `uv run play <tâche> --checkpoint-file logs/rsl_rl/<exp>/<date>/model_N.pt --viewer viser --num-envs 4`
- vidéo : `--video True --video-length 150 --video-width 1280 --video-height 720` (dans `<run>/videos/play/`)
- Les booléens de la ligne de commande (tyro) prennent une valeur explicite : `--video True`, pas `--video`.

## Pièges connus (déjà résolus, ne pas défaire)
- **Versions** : BAM (`better-actuator-models[mjlab]`, commit e9a619d, même que le simulateur officiel) impose mjlab 1.3,
  MuJoCo 3.7, mujoco-warp 3.7, warp 1.12, Python 3.12. Le simulateur officiel exige MuJoCo ≥ 3.11 : **ne pas importer
  `climbing_sim` ni `microban_sim`** (on relit leurs fichiers de données).
- `wandb==0.22.3` épinglé : les versions récentes cassent rsl_rl 5.0.1 (`start_method`).
- `Rhoban/mjlab_microban` : NE PAS utiliser son robot (ancien Microban de marche, sans boules). Seule inspiration pour
  la structure de tâche et la config BAM.
- `njmax=400, nconmax=128` : suffisants (max mesuré ~190 lignes de contrainte, ~45 contacts par robot). Trop petits →
  `nefc overflow` → contacts ignorés → les mains tombent. Plus grands → ~15 % de vitesse en moins pour rien.
  MuJoCo Warp vérifie les contacts dans la pose qpos0 au démarrage : le robot y est placé loin du mur et du sol
  (0 contact), sinon il faudrait des tampons bien plus grands.
- Pas de physique **4 ms**, décimation 5 (20 ms par décision), validé par check-scene. Simulateur officiel : 2 ms.
- `terrain=None` : toutes les origines d'env à 0, le mur est au même endroit dans chaque monde (normal).
- Avertissement « Actuator config matched 19 joint(s) ... 8 site(s) » : sans gravité.
- « solver/linesearch iterations limit reached » : fréquents en début d'entraînement, sans gravité.
  En revanche `nefc overflow` est grave.

## Modèle et fidélité au vrai robot
- Mur N1 : barreaux tous les 5 rangs (7,5 cm), 4 prises par barreau (colonnes 18, 21, 26, 29), rangées 2 à 87.
  Départ : mains (18,22) et (29,22), pieds (21,2) et (26,2). Les pieds partent à ~2,6 cm du sol.
- Servos BAM m6 XL330 : `kp_fw=125`, vin 7,2–7,8 V, chute de tension, `max_current=1.75`, latence 10–30 ms.
- Collisions "light" (défaut entraînement) : les 269 maillages du corps sont désactivés et remplacés par des boîtes
  mesurées sur ces maillages (tronc découpé en tranches, bras, cuisses, tibias, pieds, avant-bras dont la tige de la
  boule), plus les 4 boules et les 7 boîtes des servos. Elles touchent le mur, les prises, le sol **et les autres
  membres** (contype 3 / conaffinity 2). Paires voisines exclues (`SELF_EXCLUDES` : hanche/cuisse, tibia/pied).
  Coût : environ −11 % de vitesse ; interpénétration entre membres ≤ 4 mm sous actions aléatoires fortes.
- Prises : toutes ont les ~30 sphères de la coupe ; `mesh_holds` ajoute les 14 maillages du corps de la prise aux
  prises où passe le membre en mouvement (−19 % de vitesse, le même avec 4 ou 8 prises).
- Dans la pose de départ, la semelle touche le sol (~6 mm) : le robot repose en partie sur ses pieds.
- Écarts avec le simulateur officiel : pas de 4 ms, MuJoCo Warp au lieu de MuJoCo CPU, collisions en boîtes.

## Tâches (état actuel)
- **LeftFoot** : pied gauche de (21,2) vers (18,7), en haut à gauche ; la prise juste au-dessus (21,7) est hors de
  portée (constaté aussi à la main sur le vrai robot). Apprise (~92 % du temps dans la coupe vers 200 itérations).
- **BothFeet** (écrite par l'utilisateur) : les deux pieds montent d'un barreau, les mains restent accrochées.
  Cible du pied droit dans le code : (26,7), tout droit — à confirmer que c'est voulu.
- Récompenses type : maintien des appuis (`others_hooked`, σ 1 cm, poids 1), guidage (`*_reach`, σ 4 cm, poids 2),
  bonus (`*_hooked`, poids 5), pénalités `action_rate` −0,01 et `joint_limits` −1. Terminaisons : `limb_lost`
  (> 3 cm), `fell` (tronc −5 cm), 8 s. Les clés de récompense doivent être uniques (sinon elles s'écrasent).
- Attention : une liste de membres vide dans `balls_near_sockets` donne une moyenne NaN.

## Suite envisagée
- Une main seule pour trouver une cible atteignable, puis une fonction « au moins 3 appuis » dans `mdp.py`.
- Objectif de l'utilisateur : monter tout le mur N1 « par cœur » depuis le départ fixe (mur grandissant +
  reprise de checkpoint ; prises en boîtes si les maillages coûtent trop).
- Plan alternatif noté : carte d'atteignabilité par IK, positions de départ aléatoires, politique commandée
  (membre + vecteur vers la cible), puis vérification dans le simulateur officiel (export ONNX).
