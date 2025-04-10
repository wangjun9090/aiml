import requests
import json
import csv

# Define the initial search query
query = {
    "fields": [
        {"field": "startTime"},
        {"field": "userActions.targetUrl"},
        {"field": "city"},
        {"field": "userActions.extracted_data.text.stateCode"},
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
                            "gte": "2025-03-26T00:00:00.473Z",
                            "lte": "2025-03-31T23:59:59.473Z"
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

# Define headers (assuming these are set elsewhere in your environment)
headers = {
    "Content-Type": "application/json",
    # Add any authentication headers if required, e.g., "Authorization": "Bearer <token>"
}

# Initialize the scroll
url = 'http://5ff222d8671f414a8cdef95b2cded607.ctc-ece.optum.com:9200/_search'
response = requests.post(url, headers=headers, data=json.dumps(query))
response_data = response.json()

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

    if not scroll_data['hits']['hits']:
        break

    if '_scroll_id' not in scroll_data:
        raise Exception("Scroll ID not found in scroll response.")

    records.extend(scroll_data['hits']['hits'])
    scroll_id = scroll_data['_scroll_id']

# Define the expected fieldnames based on the query
fieldnames = [
    "startTime",
    "userActions.targetUrl",
    "city",
    "userActions.extracted_data.text.stateCode",
    "internalUserId",
    "userActions.extracted_data.text.topPriority",
    "userActions.extracted_data.text.specialneeds_option",
    "userActions.extracted_data.text.drugs_option"
]

# Convert records to list of dictionaries, handling missing fields
records_list = []
for record in records:
    fields = record.get('fields', {})
    # Create a dictionary with all fieldnames, defaulting to None if missing
    record_dict = {field: fields.get(field, [None])[0] for field in fieldnames}
    records_list.append(record_dict)

# Define the function to convert JSON records to CSV
def json_to_csv(json_records, csv_file_path, fieldnames):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in json_records:
            writer.writerow(record)

# Call the function with the full records_list
output_csv_path = "behavior_data.csv"  # Specify your desired output path
json_to_csv(records_list, output_csv_path, fieldnames)

print(f"Successfully wrote {len(records_list)} records to {output_csv_path}")
