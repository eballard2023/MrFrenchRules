import requests
import json
from requests.auth import HTTPBasicAuth


base_url = f"https://{domain}.atlassian.net/rest/api/3"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}
auth = HTTPBasicAuth(email, api_token)

# --- Functions for Jira Operations ---

def create_jira_issue(summary_text, issue_type="Task"):
    """Creates a new Jira issue and returns the issue key."""
    print(f"Creating a new Jira {issue_type}...")
    
    api_url = f"{base_url}/issue"
    payload = json.dumps({
        "fields": {
            "project": {
                "key": project_key
            },
            "summary": summary_text,
            "issuetype": {
                "name": issue_type
            }
        }
    })

    response = requests.post(api_url, data=payload, headers=headers, auth=auth)
    response.raise_for_status()

    response_json = response.json()
    issue_key = response_json["key"]
    issue_url = f"https://{domain}.atlassian.net/browse/{issue_key}"
    print(f"Success! Created {issue_type} '{issue_key}'.")
    print(f"View it at: {issue_url}\n")
    return issue_key

def update_jira_issue(issue_key, new_summary, new_description):
    """Updates the summary and description of a Jira issue."""
    print(f"Updating Jira issue '{issue_key}'...")
    
    api_url = f"{base_url}/issue/{issue_key}"
    payload = json.dumps({
        "fields": {
            "summary": new_summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": new_description
                            }
                        ]
                    }
                ]
            }
        }
    })

    response = requests.put(api_url, data=payload, headers=headers, auth=auth)
    response.raise_for_status()
    
    print(f"Success! Updated issue '{issue_key}'.\n")

def delete_jira_issue(issue_key):
    """Deletes a Jira issue permanently."""
    print(f"Deleting Jira issue '{issue_key}'...")
    
    api_url = f"{base_url}/issue/{issue_key}"

    response = requests.delete(api_url, headers=headers, auth=auth)
    response.raise_for_status()

    # A successful deletion returns a 204 No Content status code
    if response.status_code == 204:
        print(f"Success! Deleted issue '{issue_key}'.\n")
    else:
        print(f"Warning: Unexpected status code {response.status_code} for deletion.")


# --- Main Execution Flow ---
if __name__ == "__main__":
    try:
        # Step 1: Create a new task
        task_summary = "This is a new task created and managed by the script."
        created_issue_key = create_jira_issue(task_summary)

        # Step 2: Update the created task
        updated_summary = "Updated task via script - Summary changed."
        new_description = "This is a new description added during the update operation."
        update_jira_issue(created_issue_key, updated_summary, new_description)

        # Step 3: Delete the updated task
        delete_jira_issue(created_issue_key)

    except requests.exceptions.HTTPError as errh:
        print(f"\n--- HTTP Error ---")
        print(f"Status Code: {errh.response.status_code}")
        print(f"Response Body: {errh.response.text}")
        print(errh)
    except requests.exceptions.RequestException as err:
        print(f"\n--- Other Error ---")
        print(f"An error occurred: {err}")
