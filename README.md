# Share compute — prototype

Un particulier prête une part plafonnée de son GPU/CPU pour exécuter des inférences d'IA, dans un environnement isolé de ses fichiers.

## Principe

- **Une tâche, ce sont des données, pas du code** : un modèle ONNX et des tenseurs d'entrée. Ils sont exécutés par ONNX Runtime (DirectML, donc tout GPU sous Windows). Aucun code fourni par un client ne s'exécute chez le participant.
- **L'agent n'analyse jamais les données reçues** : il vérifie leur hash SHA-256 et la signature ed25519 du coordinateur, puis les transmet brutes au worker. Seul le worker, isolé, les décode.
- **Le worker tourne dans une sandbox** :
  - un **AppContainer** sans aucune capability : pas d'accès aux fichiers de l'utilisateur, pas de réseau (ni Internet, ni loopback) ;
  - un **Job Object** : plafond CPU strict, plafond RAM, aucun processus enfant, pas d'accès au presse-papiers ni au bureau, arrêt forcé si l'agent meurt.
- **Le budget est choisi par l'utilisateur** : CPU %, RAM, VRAM max, plafond GPU par duty cycle, calcul seulement en cas d'inactivité, pause sur batterie. Il est relu en continu : une modification s'applique sans redémarrage, et un job en cours est coupé en moins de 250 ms.
- **Les résultats sont vérifiés** : chaque job est calculé par N nœuds distincts, puis validé à la majorité avec une tolérance (les GPU ne sont pas déterministes au bit près). En cas de désaccord, un réplica supplémentaire est demandé, jusqu'à 5.

```
crates/protocol     types partagés, signature, comparateur, IPC agent↔worker
crates/coordinator  serveur (axum, état en mémoire) + CLI client submit/status
crates/agent        budget, boucle de travail, sandbox Windows, détection inactivité/batterie/VRAM
crates/worker       exécuteur ONNX lancé dans la sandbox (+ modes de diagnostic)
```

## Utilisation

```powershell
cargo build --release
mkdir run; cd run

# 1. Coordinateur (crée coordinator.key au premier lancement)
..\target\release\coordinator.exe serve

# 2. Un agent par participant (autre terminal)
..\target\release\agent.exe --config agent1.toml init --name node1
..\target\release\agent.exe --config agent1.toml config --cpu 25 --ram 2048 --vram 2048 --gpu-duty 50
..\target\release\agent.exe --config agent1.toml test-sandbox     # facultatif : vérifie l'isolation
..\target\release\agent.exe --config agent1.toml run

# 3. Côté client (depuis la racine du dépôt)
$id = .\target\release\coordinator.exe submit --model models\mobilenetv2-12.onnx --image samples\dog.jpg --replicas 2
.\target\release\coordinator.exe status $id --wait
```

Options de `agent config` : `--cpu`, `--ram`, `--vram`, `--gpu-duty`, `--gpu`, `--idle-only`, `--idle-seconds`, `--pause-on-battery`, `--sandbox`.

Au premier lancement, `agent run` donne aux AppContainers (`S-1-15-2-1`) le droit de lire et d'exécuter le dossier du worker, pour qu'il puisse charger ses DLL. Sous `Program Files`, ce droit existe déjà.

## Résultats mesurés (RTX 2060, Windows 11)

| Test | Résultat |
|---|---|
| Lister les Documents, lire `agent1.toml` (qui contient le token), écrire dans `C:\Windows\Temp` | refusé (accès refusé) |
| Se connecter à 1.1.1.1:443 ou au coordinateur sur 127.0.0.1 | bloqué |
| Lancer `cmd.exe` | bloqué (quota du Job Object) |
| Plafond CPU à 25 %, avec 6 threads qui tournent à fond | 27,9 % mesurés |
| Plafond RAM à 2048 Mo | allocation refusée à 2033 Mo |
| Classification sur 2 nœuds (DirectML), `replicas=2` | validée ; Samoyède à 83 %, chat (tiger/Egyptian cat) |
| Utilisateur actif pendant un job | worker tué en 184 ms, job repris par un autre nœud |
| Puissance GPU au-dessus du repos, plafond à 100 / 50 / 20 % | +28,5 / +17,3 / +4,9 W |

Le pourcentage « utilisation » de `nvidia-smi` n'est pas un bon indicateur du plafond GPU. Le GPU baisse ses fréquences entre deux inférences, donc chacune dure plus longtemps. La consommation électrique est le bon indicateur.

## Limites connues

- **On ne peut pas cacher le calcul au propriétaire du PC.** Avec les droits admin, il peut lire la mémoire du worker. Sur du matériel grand public, rien ne l'empêche (le *confidential computing* n'existe que sur serveur). Il faut donc choisir des workloads dont les données ne sont pas sensibles.
- Le coordinateur garde tout en mémoire, et n'a ni TLS, ni authentification client, ni paiement.
- La clé du coordinateur est épinglée à la première connexion (*trust on first use*). En production, elle serait livrée avec l'installeur.
- Le plafond VRAM se base sur l'estimation déclarée dans le job. Il n'y a pas de limite matérielle de VRAM par processus.
- Il n'y a pas encore d'icône dans la barre des tâches.
- Une majorité de nœuds malveillants qui s'entendent sur le même faux résultat passerait la vérification. Une vraie plateforme ajouterait des tâches pièges dont la réponse est connue, et un score de réputation.
