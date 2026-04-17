# Intan Trigger Plotter

Programme Python pour lire des fichiers Intan `.rhs`, detecter un **front montant ou descendant** sur `ANALOG_IN 0`, extraire des fenetres temporelles autour de chaque trigger, puis enregistrer la moyenne par canal dans **un seul PDF multi-pages** (aucune fenetre matplotlib).

## Structure du projet

- `src/core.py` : lecture RHS et calculs
- `src/plotting.py` : affichage et export PDF
- `src/gui.py` : interface graphique Qt (selection fichier + parametres)
- `src/cli.py` : point d'entree ligne de commande
- `src/load_intan_rhs_format.py` : lecteur Intan RHS
- `run_gui.py` : lanceur Python simple pour la GUI

## Prerequis

1. Installer Python 3.9+
2. Installer les dependances:

```bash
pip install -r requirements.txt
```

3. Le lecteur Intan `load_intan_rhs_format.py` doit etre dans `src/`.

## Utilisation

### Mode GUI (selection du fichier RHS)

```bash
python src/cli.py --gui --save-dir "plots"
```

L'interface comporte deux **onglets** :
- **Analyse** : un fichier `.rhs`, moyenne par canal, PDF nomme comme le `.rhs`.
- **Comparaison** : deux fichiers `.rhs`, memes parametres (bloc commun sous les onglets), courbes **superposees** par canal dans un PDF `{nom1}_vs_{nom2}.pdf`.

Parametres communs (seuil, front, pre/post, passe-bas, dossier PDF) :
- type de front sur ANALOG_IN 0 (descendant ou montant)
- pre/post trigger
- dossier de sortie du PDF (optionnel : si vide, **meme dossier que le premier .rhs** en analyse simple, ou enregistrement 1 en comparaison)

En analyse simple, le PDF porte le **meme nom de base** que le fichier `.rhs` (ex. `session.rhs` → `session.pdf`).

### Mode ligne de commande

```bash
python src/cli.py "session01.rhs" --save-dir "plots"
```

### Lanceur Python GUI

```bash
python run_gui.py
```

## Options principales

- `--edge` : `falling` (descendant) ou `rising` (montant) sur ANALOG_IN 0 (defaut: `falling`)
- `--threshold` : seuil de comparaison pour le front (defaut: `1.0`)
- `--pre` : secondes avant trigger (defaut: `2.0`)
- `--post` : secondes apres trigger (defaut: `10.0`)
- `--save-dir` : dossier de sortie du PDF (defaut : dossier du fichier `.rhs`)
- `--lowpass-hz` : frequence de coupure (Hz) d'un **passe-bas Butterworth** (ordre 4, `filtfilt`) sur les canaux amplificateur ; omis = pas de filtre

Les courbes amplificateur sont en **microvolts (µV)** (convention du lecteur Intan).

Le PDF est en multi-pages (une page par canal), nomme comme le fichier RHS : `<nom_du_fichier_sans_extension>.pdf`.
