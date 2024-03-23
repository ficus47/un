import os

def limiter_elements_sous_dossiers(dossier_racine, X):
    """
    Parcourt chaque sous-dossier du dossier racine et limite le nombre d'éléments à X.
    Args:
        dossier_racine (str): Chemin du dossier racine à parcourir.
        X (int): Nombre maximal d'éléments à conserver dans chaque sous-dossier.
    """
    for dossier_actuel, sous_dossiers, fichiers in os.walk(dossier_racine):
        for sous_dossier in sous_dossiers:
            chemin_sous_dossier = os.path.join(dossier_actuel, sous_dossier)
            elements_dans_sous_dossier = os.listdir(chemin_sous_dossier)
            elements_a_conserver = elements_dans_sous_dossier[:X]

            # Supprimer les éléments excédentaires
            for element in elements_dans_sous_dossier:
                chemin_element = os.path.join(chemin_sous_dossier, element)
                if element not in elements_a_conserver:
                    os.remove(chemin_element)

            print(f"Éléments dans {sous_dossier} limités à {X} éléments.")

# Exemple d'utilisation
dossier_racine = "NULL"
x = 40
#limiter_elements_sous_dossiers(dossier_racine, x)


import hashlib
import os

def supprimer_photos_identiques(dossier):
  """
  Supprime toutes les photos identiques dans un dossier donné.

  Args:
      dossier (str): Le chemin du dossier à analyser.
  """
  fichiers = os.listdir(dossier)
  fichiers_a_supprimer = []

  # Dictionnaire pour stocker les empreintes digitales et les fichiers correspondants
  empreintes_digitales = {}

  for fichier in fichiers:
    chemin_fichier = os.path.join(dossier, fichier)

    if not os.path.isfile(chemin_fichier):
      continue

    # Calculer l'empreinte digitale du fichier
    with open(chemin_fichier, 'rb') as f:
      empreinte_digitale = hashlib.md5(f.read()).hexdigest()

    if empreinte_digitale in empreintes_digitales:
      # Si l'empreinte digitale existe déjà, le fichier est un doublon
      fichiers_a_supprimer.append(chemin_fichier)
    else:
      # Ajouter l'empreinte digitale et le fichier au dictionnaire
      empreintes_digitales[empreinte_digitale] = chemin_fichier

  # Supprimer les fichiers doublons
  for fichier_a_supprimer in fichiers_a_supprimer:
    os.remove(fichier_a_supprimer)

if __name__ == "__main__":
  dossier = "young"  # Remplacez par le chemin réel de votre dossier
  supprimer_photos_identiques(dossier)
