import requests
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
base_url = "https://localhost:4999/v1/api/"
account_id = "U14993841"

def suppress_order_reply(message_ids):
    """
    Suppresses specified order reply messages for the duration of the brokerage session.
    For example, to suppress the "Mandatory Cap Price" confirmation message (ID "o10153").
    """
    endpoint = "iserver/questions/suppress"
    url = base_url + endpoint
    payload = {"messageIds": message_ids}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=payload, headers=headers, verify=False)
    print("Suppress Status Code:", response.status_code)
    try:
        print("Suppress Response JSON:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Suppress Response Text:", response.text)
        

def place_order():
    
    endpoint = f"iserver/account/{account_id}/orders"
    url = base_url + endpoint

    order_data = {
        "orders": [
            {
                "acctId": account_id,
                "conid": 265598,
                "secType": "265598:STK",
                "cOID": "ACH1-LMT-BUY",  
                "orderType": "LMT",      
                "listingExchange": "SMART",  
                "side": "BUY",
                "tif": "DAY",            
                "quantity": 2,
                "ticker": "ACHV",
                "price": 3.00       
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=order_data, headers=headers, verify=False)
    
    # Print out status and response for debugging
    print("Status Code:", response.status_code)
    try:
        resp_json = response.json()
        print("Response JSON:\n", json.dumps(resp_json, indent=2))
    except Exception as e:
        print("Response Text:\n", response.text)

if __name__ == "__main__":
    suppress_order_reply(["o10153"])
    place_order()
    
