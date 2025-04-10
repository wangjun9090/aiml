import requests
import json
import csv

# Define the initial search query with scroll parameter
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
    "scroll": "1m",
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

# Define headers (adjust with your authentication)
headers = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer <your-token>"
}

# Initialize the scroll
url = 'http://5ff222d8671f414a8cdef95b2cded607.ctc-ece.optum.com:9200/_search'
response = requests.post(url, headers=headers, data=json.dumps(query))
response_data = response.json()

# Debug: Print the initial response
print("Initial response:", json.dumps(response_data, indent=2))

# Check for scroll ID
records = response_data.get('hits', {}).get('hits', [])
if not records:
    raise Exception("No hits found in initial response.")

if '_scroll_id' in response_data:
    scroll_id = response_data['_scroll_id']
    print(f"Scroll ID found: {scroll_id}, Initial hits: {len(records)}")
else:
    print("Scroll ID not found. Processing initial response only.")
    scroll_id = None

# Fetch subsequent batches if scrolling is supported
scroll_url = 'http://5ff222d8671f414a8cdef95b2cded607.ctc-ece.optum.com:9200/_search/scroll'
if scroll_id:
    while True:
        scroll_query = {
            "scroll": "1m",
            "scroll_id": scroll_id
        }
        scroll_response = requests.post(scroll_url, headers=headers, data=json.dumps(scroll_query))
        scroll_data = scroll_response.json()

        if not scroll_data.get('hits', {}).get('hits', []):
            break

        if '_scroll_id' not in scroll_data:
            raise Exception("Scroll ID not found in scroll response.")

        records.extend(scroll_data['hits']['hits'])
        scroll_id = scroll_data['_scroll_id']

# Define the expected fieldnames
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

# Convert records to list of dictionaries, preserving all values
records_list = []
for record in records:
    fields = record.get('fields', {})
    # Join multi-valued fields into a single string with a separator
    record_dict = {
        field: ", ".join(fields.get(field, [None]) if fields.get(field) else [None])
        for field in fieldnames
    }
    records_list.append(record_dict)

# Define the function to convert JSON records to CSV
def json_to_csv(json_records, csv_file_path, fieldnames):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in json_records:
            writer.writerow(record)

# Call the function with the full records_list
output_csv_path = "behavior_data.csv"
json_to_csv(records_list, output_csv_path, fieldnames)

print(f"Successfully wrote {len(records_list)} records to {output_csv_path}")
