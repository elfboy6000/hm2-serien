import numpy as np

def lagrange_int(x, y, x_int):
    """
    Führt eine Lagrange-Interpolation durch.

    Args:
        x (list or np.array): Liste der x-Koordinaten der bekannten Datenpunkte.
        y (list or np.array): Liste der y-Koordinaten der bekannten Datenpunkte.
        x_int (float): Die x-Koordinate, an der interpoliert werden soll.

    Returns:
        float: Der interpolierte y-Wert (y_int).
    """
    x_points = np.array(x)
    y_points = np.array(y)
    
    # Entferne NaN-Werte aus den Datenpunkten
    valid_indices = ~np.isnan(y_points)
    x_known = x_points[valid_indices]
    y_known = y_points[valid_indices]
    
    n = len(x_known)
    y_int = 0.0
    
    for j in range(n):
        # Berechne das Lagrange-Basispolynom L_j(x)
        p = 1.0
        for i in range(n):
            if i != j:
                p *= (x_int - x_known[i]) / (x_known[j] - x_known[i])
        
        # Addiere den Beitrag zum interpolierten Wert
        y_int += y_known[j] * p
        
    return y_int

# Gegebene Datenpunkte
hoehe = [0, 2500, 3750, 5000, 10000]
druck = [1013, 747, np.nan, 540, 226]

# Die zu interpolierende Höhe
hoehe_zu_interpolieren = 3750

# Führe die Lagrange-Interpolation durch, um den Druck zu finden
interpolierter_druck = lagrange_int(hoehe, druck, hoehe_zu_interpolieren)

print(f"Gegebene Höhen [m]: {hoehe}")
print(f"Gegebene Drücke [hPa]: {druck}")
print("-" * 30)
print(f"Der interpolierte Atmosphärendruck auf einer Höhe von {hoehe_zu_interpolieren} m beträgt: {interpolierter_druck:.2f} hPa")
