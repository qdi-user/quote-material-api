import requests

def fetch_material_details(recyclable, finish, opacity):
    """
    Fetches material details based on the provided parameters.
    """
    url = "https://quote-material-api.onrender.com/query-materials"
    params = {
        'recyclable': recyclable.lower(),  # Ensure lowercase for consistency
        'finish': finish.lower(),
        'opacity': opacity.lower()
    }

    try:
        response = requests.post(url, params=params)
        
        if response.status_code == 200:
            result_data = response.json().get('Result')
            if result_data and len(result_data) > 0:
                return result_data[0]
            else:
                return {"error": "No material details found for the given criteria."}
        else:
            return {"error": f"Unable to fetch material details. Status code: {response.status_code}"}

    except requests.RequestException as e:
        return {"error": f"Request error: {str(e)}"}
