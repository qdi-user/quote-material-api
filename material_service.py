import requests
import logging

def fetch_material_details(recyclable, finish, opacity):
    """
    Fetches material details based on the provided parameters.
    Returns all matching materials.
    """
    url = "https://quote-material-api.onrender.com/query-materials"
    
    # Convert to JSON body instead of params
    payload = {
        'Recyclable': recyclable.lower(),  # Use capitalized key to match the input model
        'Finish': finish.lower(),
        'Opacity': opacity.lower()
    }

    try:
        # Log the outgoing request for debugging
        logging.info(f"Sending request to {url} with payload: {payload}")
        
        response = requests.post(url, json=payload)
        
        # Log the raw response for debugging
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.text}")
        
        if response.status_code == 200:
            result_data = response.json()  # Get the full list of results
            
            if result_data and len(result_data) > 0:
                return result_data  # Return ALL matching materials
            else:
                return {"error": "No material details found for the given criteria."}
        else:
            return {
                "error": f"Unable to fetch material details.",
                "status_code": response.status_code,
                "response_text": response.text
            }

    except requests.RequestException as e:
        logging.error(f"Request error: {str(e)}")
        return {"error": f"Request error: {str(e)}"}