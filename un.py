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
x = 60
limiter_elements_sous_dossiers(dossier_racine, x)
