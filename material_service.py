import requests

def fetch_material_details(recyclable, finish, opacity):
    """
    Fetches material details based on the provided parameters.
    """
    url = "https://quote-material-api.onrender.com/query-materials"
    
    # Convert to JSON body instead of params
    payload = {
        'Recyclable': recyclable.lower(),  # Use capitalized key to match the input model
        'Finish': finish.lower(),
        'Opacity': opacity.lower()
    }

    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result_data = response.json()  # Directly access the returned list
            if result_data and len(result_data) > 0:
                return result_data[0]
            else:
                return {"error": "No material details found for the given criteria."}
        else:
            return {"error": f"Unable to fetch material details. Status code: {response.status_code}"}

    except requests.RequestException as e:
        return {"error": f"Request error: {str(e)}"}