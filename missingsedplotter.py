from common_functions import RVVD_PATH, BASE_PATH
import os
import pandas as pd


table = pd.read_csv("solved_orbit_tracker.txt", delimiter=",")

for ind, row in table.iterrows():
    if row["solved"] == "y":
        if not os.path.isfile(RVVD_PATH+ f"/SEDs/{row['gaia_id']}/photometry.sl") or (not os.path.isfile(RVVD_PATH+ f"/models/{row['gaia_id']}/spectroscopy.sl") and not os.path.isdir(BASE_PATH+f"/spektralfits/{row['gaia_id']}")):
            print(row["gaia_id"])
