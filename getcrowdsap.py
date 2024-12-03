import argparse
import lightkurve as lk
from astroquery.vizier import Vizier


def get_tic(gaia_id: int) -> str:
    """
    Query the IV/39/tic82 catalog from Vizier using a Gaia ID to retrieve the TIC ID.

    Parameters:
    - gaia_id (int): The Gaia ID to query.

    Returns:
    - str: The TIC ID if found, or a message indicating no match was found.
    """
    # Define the catalog and the columns to retrieve
    catalog = "IV/39/tic82"
    columns = ["GAIA", "TIC"]

    # Initialize Vizier with a row limit and column selection
    Vizier.ROW_LIMIT = -1  # No row limit
    Vizier.columns = columns

    # Perform the query
    try:
        result = Vizier.query_constraints(catalog=catalog, GAIA=str(gaia_id))
        if result:
            # Extract the TIC ID from the first table's results
            tic_table = result[0]
            tic_id = tic_table["TIC"][0]  # Assume first match is sufficient
            return str(tic_id)
        else:
            return "No TIC found for Gaia ID: {}".format(gaia_id)
    except Exception as e:
        return f"Error during query: {str(e)}"


def get_crowdsap_lightkurve(tic_id):
    if len(str(tic_id)) > 10:
        tic_id = get_tic(tic_id)

    # Search for the light curve
    search_result = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS')

    # Check if any light curves were found
    if len(search_result) == 0:
        return f"No light curves found for TIC {tic_id}."

    # Download the light curve file
    lc = search_result.download()

    # Access the header information
    header = lc.meta

    # Extract the CROWDSAP value
    crowdsap = header.get('CROWDSAP', 'CROWDSAP not found')
    return crowdsap


def main():
    parser = argparse.ArgumentParser(description='Retrieve CROWDSAP value for a given TIC ID using lightkurve.')
    parser.add_argument('tic_id', type=str, help='TIC ID of the target')
    args = parser.parse_args()

    crowdsap = get_crowdsap_lightkurve(args.tic_id)
    print(f"CROWDSAP: {crowdsap}")


if __name__ == '__main__':
    main()