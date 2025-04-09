import requests
import json
import csv

# Define the Elasticsearch endpoint and headers
url = 'http://5ff222d8671f414a8cdef95b2cded607.ctc-ece.optum.com:9200/user_sessions/_search?scroll=1m'
headers = {
    'Authorization': 'Basic ZWxhc3RpYzpDZ1pOb3lIU3ZLdVZKa0lySWh0YWg5SzA=',
    'Content-Type': 'application/json'
}

# Define the initial search query (removed "scroll": "1m")
query = {
    "fields": [
        {"field": "startTime"},
        {"field": "userActions.targetUrl"},
        {"field": "internalUserId"},
        {"field": "userActions.extracted_data.text.topPriority"},
        {"field": "userActions.extracted_data.text.specialneeds_option"},
        {"field": "userActions.extracted_data.text.drugs_option"}
    ],
    "size": 1000,
    "_source": False,
    "query": {
        "bool": {
            "filter": [
                {
                    "range": {
                        "startTime": {
                            "format": "strict_date_optional_time",
                            "gte": "2025-03-05T00:00:00.473Z",
                            "lte": "2025-03-06T23:59:59.473Z"
                        }
                    }
                },
                {"exists": {"field": "internalUserId"}}
            ],
            "must_not": [
                {"match_phrase": {"internalUserId": "null"}},
                {"match_phrase": {"internalUserId": "undefined"}}
            ]
        }
    }
}

# Initialize the scroll
response = requests.post(url, headers=headers, data=json.dumps(query))
response_data = response.json()
print("Initial response:", json.dumps(response_data, indent=2))  # Debug
if '_scroll_id' not in response_data:
    raise Exception("Scroll ID not found in initial response.")
scroll_id = response_data['_scroll_id']
records = response_data['hits']['hits']

# Fetch subsequent batches
scroll_url = 'http://5ff222d8671f414a8cdef95b2cded607.ctc-ece.optum.com:9200/_search/scroll'
while True:
    scroll_query = {
        "scroll": "1m",
        "scroll_id": scroll_id
    }
    scroll_response = requests.post(scroll_url, headers=headers, data=json.dumps(scroll_query))
    scroll_data = scroll_response.json()
    print("Scroll response:", json.dumps(scroll_data, indent=2))  # Debug
    if not scroll_data['hits']['hits']:
        break
    if '_scroll_id' not in scroll_data:
        raise Exception("Scroll ID not found in scroll response.")
    records.extend(scroll_data['hits']['hits'])
    scroll_id = scroll_data['_scroll_id']

# Convert records to list of dictionaries
records_list = [record['fields'] for record in records]

# Define the function to convert JSON records to CSV
def json_to_csv(json_records, csv_file_path):
    fieldnames = json_records[0].keys()
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in json_records:
            writer.writerow(record)

# Save to CSV
csv_file_path = r'c:\Users\jwang46\Documents\MR_AI\Website\persona\elastic_us_0301_0331_2025.csv'
json_to_csv(records_list, csv_file_path)
print("Data saved to CSV successfully.")
