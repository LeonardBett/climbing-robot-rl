### `climb_env_cfg.py` : les règles du jeu

Ce fichier ne simule rien : ses fonctions **renvoient une description** de l'environnement, que mjlab lit pour
construire les milliers de robots sur le GPU. Chaque règle suit le même motif :

```python
RewardTermCfg(func=mdp.limb_hooked, weight=5.0, params={"sockets": goal, "limb": LEFT_FOOT})
```

À chaque pas, mjlab appelle `func(env, **params)` pour tous les robots à la fois et multiplie le résultat par `weight`.

- **`_base_cfg`** : ce qui est commun à toutes les tâches.
  - *Scène* : le robot (`robot.py`) et le mur (`scene.py`), tous les robots au même endroit, chacun dans son monde.
  - *Actions* : 18 consignes d'angle (pas la tête), consigne = pose accrochée + 0,5 × action. BAM transforme ensuite
    la consigne en couple réaliste (couple limité, chute de tension, latence).
  - *Observations* : l'**actor** ne voit que ce que le vrai robot mesure (gyroscope, gravité dans le tronc, angles et
    vitesses des servos, dernière action ; 60 valeurs), avec bruit et retards. Le **critic** voit en plus la vérité
    terrain (position de chaque boule par rapport à sa coupe cible ; 79 valeurs), car il ne sert qu'à l'entraînement.
  - *Événements* : `bam_init` au démarrage (une copie des frottements des servos par robot), retour à la pose
    accrochée au début de chaque épisode.
  - *Terminaisons* : temps écoulé, chute (tronc descendu de plus de 5 cm), NaN.
  - *Pénalités* : saccades (`action_rate`), articulations en butée (`joint_limits`).
  - *Physique* : pas de 4 ms, une décision toutes les 20 ms comme le vrai robot.
- **Une fonction par tâche** (`hang_env_cfg`, `left_foot_env_cfg`, `both_feet_env_cfg`) : elle prend la base, choisit
  les prises cibles, ajoute ses récompenses et ses terminaisons.
- **`ppo_cfg`** : les réglages de PPO, repris de la tâche de marche de Rhoban (`mjlab_microban`).

Les récompenses suivent le schéma classique contre les récompenses rares : une **récompense dense** qui guide
(`foot_reach`, une cloche selon la distance à la cible), un **bonus** quand le but est atteint (`foot_hooked`, boule à
moins de 6 mm du centre de la coupe), une **contrainte douce** sur les appuis (`others_hooked`) et une **contrainte
dure** qui coupe l'épisode s'ils lâchent (`limb_lost`, plus de 3 cm).

### `mdp.py` : la boîte à outils de l'escalade

De petites fonctions de quelques lignes, chacune calculant une seule chose pour tous les robots à la fois (tenseurs
PyTorch sur le GPU). Les membres sont désignés par leur numéro : 0 main gauche, 1 main droite, 2 pied gauche,
3 pied droit. « Accroché » a le même sens que `hooked()` du simulateur officiel : centre de la boule à moins de 6 mm
du centre de la coupe.

| Fonction | Rôle | Utilisée comme |
|---|---|---|
| `ball_positions`, `ball_distances` | positions des 4 boules, distance de chacune à sa coupe cible | brique de base |
| `ball_offsets` | vecteur boule → coupe cible pour les 4 membres (12 valeurs) | observation du critic |
| `hooked_flags` | chaque boule accrochée ou non (4 valeurs) | observation du critic |
| `balls_near_sockets` | note de 0 à 1 par boule selon sa distance (cloche de largeur σ), moyenne sur les membres choisis | récompense (guidage, maintien) |
| `balls_hooked` | fraction des membres choisis accrochés | récompense (bonus) |
| `limb_reach` | cloche de distance pour un seul membre vers sa cible | récompense (guidage) |
| `limb_hooked` | 1 si ce membre est accroché à sa cible | récompense (bonus) |
| `trunk_dropped` | le tronc est descendu sous sa hauteur de départ | terminaison (chute) |
| `limbs_lost` | un membre d'appui s'est éloigné de sa coupe | terminaison (appui perdu) |

On ajoute une fonction ici seulement quand aucune ne mesure ce qu'on veut récompenser ; on l'utilise ensuite comme
règle dans `climb_env_cfg.py`.
