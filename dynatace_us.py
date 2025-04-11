import requests
import json
import csv
from datetime import datetime

# Define the Elasticsearch endpoint and headers
url = 'http://5ff222d8671f414a8cdef95b2cded607.ctc-ece.optum.com:9200/user_sessions/_search?scroll=1m'
headers = {
    'Authorization': 'Basic ZWxhc3RpYzpDZ1pOb3lIU3ZLdVZKa0lySWh0YWg5SzA=',
    'Content-Type': 'application/json'
}

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
                            "gte": "2025-03-01T00:00:00.473Z",
                            "lte": "2025-03-02T23:59:59.473Z"
                        }
                    }
                },
                {"exists": {"field": "internalUserId"}},
                {
                    "bool": {
                        "must": [],
                        "filter": [
                            {
                                "match_phrase": {
                                    "custom_user_type.keyword": "REAL_USER"
                                }
                            },
                            {
                                "match_phrase": {
                                    "supported_browser": True
                                }
                            },
                            {
                                "bool": {
                                    "minimum_should_match": 1,
                                    "should": [
                                        {
                                            "match_phrase": {
                                                "application.keyword": "Online - aarpmedicareplans.com"
                                            }
                                        },
                                        {
                                            "match_phrase": {
                                                "application.keyword": "Online - uhcmedicaresolutions.com"
                                            }
                                        }
                                    ]
                                }
                            }
                        ],
                        "should": [],
                        "must_not": []
                    }
                }
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

# Define the input fieldnames
input_fieldnames = [
    "startTime",
    "userActions.targetUrl",
    "city",
    "userActions.extracted_data.text.stateCode",
    "internalUserId",
    "userActions.extracted_data.text.topPriority",
    "userActions.extracted_data.text.specialneeds_option",
    "userActions.extracted_data.text.drugs_option"
]

# Define the output fieldnames (rename internalUserId to useId)
output_fieldnames = [
    "startTime",
    "userActions.targetUrl",
    "city",
    "userActions.extracted_data.text.stateCode",
    "useId",
    "userActions.extracted_data.text.topPriority",
    "userActions.extracted_data.text.specialneeds_option",
    "userActions.extracted_data.text.drugs_option"
]

# Define the allowed URL patterns
allowed_url_patterns = [
    "health-plans/plan-summary",
    "/plan-compare",
    "health-plans/details.html",
    "site-search.html?q1",
    "additionalBenefits="
]

# Convert records to list of dictionaries and preprocess startTime
records_list = []
for record in records:
    fields = record.get('fields', {})
    record_dict = {}
    for field in input_fieldnames:
        if field == "startTime":
            # Convert startTime to yyyy-mm-dd
            start_time_values = fields.get(field, [""])
            if start_time_values and start_time_values[0]:
                try:
                    dt = datetime.strptime(start_time_values[0], "%Y-%m-%dT%H:%M:%S.%fZ")
                    record_dict[field] = dt.strftime("%Y-%m-%d")
                except ValueError:
                    record_dict[field] = ""
            else:
                record_dict[field] = ""
        elif field == "userActions.targetUrl":
            # Get URLs, filter by allowed patterns, keep as list for now
            urls = fields.get(field, [])
            filtered_urls = [
                str(url) for url in urls
                if url is not None and any(pattern in str(url) for pattern in allowed_url_patterns)
            ]
            record_dict[field] = filtered_urls  # Store as list for grouping
        elif field == "userActions.extracted_data.text.stateCode":
            # Split stateCode string into list for grouping
            state_codes = fields.get(field, [])
            record_dict[field] = [str(code) for code in state_codes if code is not None]
        else:
            # Handle other fields
            record_dict[field] = ", ".join(str(item) if item is not None else "" for item in fields.get(field, []))
        # Ensure UTF-8 compatibility
        if isinstance(record_dict[field], str):
            record_dict[field] = record_dict[field].encode('utf-8', errors='replace').decode('utf-8')
    records_list.append(record_dict)

# Group by internalUserId and startTime (date only), union and dedupe userActions.targetUrl and stateCode
grouped_records = {}
for record in records_list:
    key = (record["internalUserId"], record["startTime"])
    if key not in grouped_records:
        grouped_records[key] = {
            "startTime": record["startTime"],
            "userActions.targetUrl": set(record["userActions.targetUrl"]),  # Use set for deduplication
            "city": record["city"],
            "userActions.extracted_data.text.stateCode": set(record["userActions.extracted_data.text.stateCode"]),  # Use set for deduplication
            "internalUserId": record["internalUserId"],
            "userActions.extracted_data.text.topPriority": record["userActions.extracted_data.text.topPriority"],
            "userActions.extracted_data.text.specialneeds_option": record["userActions.extracted_data.text.specialneeds_option"],
            "userActions.extracted_data.text.drugs_option": record["userActions.extracted_data.text.drugs_option"]
        }
    else:
        # Union URLs and deduplicate
        grouped_records[key]["userActions.targetUrl"].update(record["userActions.targetUrl"])
        # Union stateCodes and deduplicate
        grouped_records[key]["userActions.extracted_data.text.stateCode"].update(record["userActions.extracted_data.text.stateCode"])
        # For other fields, keep first non-empty value or combine if desired
        for field in input_fieldnames:
            if field not in ["startTime", "userActions.targetUrl", "userActions.extracted_data.text.stateCode", "internalUserId"]:
                if not grouped_records[key][field] and record[field]:
                    grouped_records[key][field] = record[field]
                elif record[field] and grouped_records[key][field] != record[field]:
                    # Optional: Combine non-empty values (e.g., comma-separated)
                    grouped_records[key][field] = f"{grouped_records[key][field]}, {record[field]}" if grouped_records[key][field] else record[field]

# Convert grouped records to final list, join URLs and stateCodes, rename internalUserId, and filter out empty userActions.targetUrl
final_records = []
for key, record in grouped_records.items():
    record_dict = record.copy()
    # Convert URL set to sorted list and join
    record_dict["userActions.targetUrl"] = ", ".join(sorted(record["userActions.targetUrl"])) if record["userActions.targetUrl"] else ""
    # Convert stateCode set to sorted list and join
    record_dict["userActions.extracted_data.text.stateCode"] = ", ".join(sorted(record["userActions.extracted_data.text.stateCode"])) if record["userActions.extracted_data.text.stateCode"] else ""
    # Skip records where userActions.targetUrl is empty
    if not record_dict["userActions.targetUrl"]:
        continue
    # Rename internalUserId to useId
    record_dict["useId"] = record_dict.pop("internalUserId")
    final_records.append(record_dict)

# Define the function to convert JSON records to CSV
def json_to_csv(json_records, csv_file_path, fieldnames):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in json_records:
            writer.writerow(record)

# Save to CSV
csv_file_path = r'c:\Users\jwang46\Documents\MR_AI\Website\persona\data\elastic_us_0301_0302_2025.csv'
json_to_csv(final_records, csv_file_path, output_fieldnames)
print("Data saved to CSV successfully.")
