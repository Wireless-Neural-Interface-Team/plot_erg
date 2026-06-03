# Intan Trigger Plotter

Programme Python pour lire des fichiers Intan `.rhs`, détecter un **front montant ou descendant** sur `ANALOG_IN 0`, extraire des fenêtres temporelles autour de chaque trigger, puis enregistrer la moyenne par canal dans **un seul PDF multi-pages** (aucune fenêtre matplotlib).

## Structure du projet

- `src/core.py` : lecture RHS, filtrage Intan RHX, triggers, piles mmap
- `src/plotting.py` : export PDF multi-panneaux (moyennes, RMS, raster, PSTH, ISI)
- `src/gui.py` : interface graphique Qt (sélection fichier + paramètres)
- `src/cli.py` : point d'entrée ligne de commande
- `src/load_intan_rhs_format.py` : lecteur Intan RHS
- `run_gui.py` : lanceur Python simple pour la GUI

## Prérequis

1. Installer Python 3.9+
2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

3. Le lecteur Intan `load_intan_rhs_format.py` doit être dans `src/`.

## Utilisation

### Mode GUI (sélection du fichier RHS)

```bash
python src/cli.py --gui --save-dir "plots"
```

L'interface comporte deux **onglets** :
- **Analyse** : un ou plusieurs fichiers `.rhs`, PDF par canal avec moyennes superposées.
- **Comparaison** : plusieurs enregistrements, mêmes paramètres, courbes **superposées** par canal.

Paramètres communs (seuil, front, pre/post, filtre Intan RHX, dossier PDF) :
- type de front sur ANALOG_IN 0 (descendant, montant, ou aucun pour sections fixes)
- fenêtre pre/post trigger
- filtre Intan pour moyennes filtrées, RMS et panneaux spike (passe-haut ou passe-bas, Bessel/Butterworth, ordre, coupure)
- dossier de sortie du PDF (optionnel : si vide, **même dossier que le premier .rhs**)

### Mode ligne de commande

```bash
python src/cli.py "session01.rhs" --save-dir "plots"
```

### Lanceur Python GUI

```bash
python run_gui.py
```

## Options principales

- `--edge` : `falling`, `rising` ou `none` sur ANALOG_IN 0 (défaut : `falling`)
- `--threshold` : seuil de détection du front (défaut : `1.0`)
- `--pre` / `--post` : secondes avant/après trigger
- `--save-dir` : dossier de sortie du PDF
- `--intan-spike-filter` : `highpass` ou `lowpass` (défaut : `highpass`)
- `--intan-filter-type` : `bessel` ou `butterworth`
- `--intan-filter-order` : ordre du filtre Intan (1–8)
- `--intan-filter-cutoff-hz` : fréquence de coupure (Hz)

Les courbes amplificateur sont en **microvolts (µV)**. Le PDF contient, par canal : moyenne brute, moyenne filtrée Intan, premier trigger (HP et brut), RMS, raster, PSTH, taux par trial et ISI.
