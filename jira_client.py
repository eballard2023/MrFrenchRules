import requests
import json
import os
from requests.auth import HTTPBasicAuth

# Jira configuration from environment
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN", "")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "")

class JiraClient:
    def __init__(self):
        self.domain = JIRA_DOMAIN
        self.email = JIRA_EMAIL
        self.api_token = JIRA_API_TOKEN
        self.project_key = JIRA_PROJECT_KEY
        
        self.base_url = f"https://{self.domain}.atlassian.net/rest/api/3"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.auth = HTTPBasicAuth(self.email, self.api_token)
    
    def create_task(self, summary_text: str, description: str = None) -> str:
        """Creates a new Jira task and returns the issue key."""
        try:
            api_url = f"{self.base_url}/issue"
            
            fields = {
                "project": {"key": self.project_key},
                "summary": summary_text,
                "issuetype": {"name": "Task"}
            }
            
            if description:
                fields["description"] = {
                    "type": "doc",
                    "version": 1,
                    "content": [{
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}]
                    }]
                }
            
            payload = json.dumps({"fields": fields})
            
            response = requests.post(api_url, data=payload, headers=self.headers, auth=self.auth)
            response.raise_for_status()
            
            response_json = response.json()
            issue_key = response_json["key"]
            print(f"Successfully created Jira task: {issue_key}")
            return issue_key
            
        except Exception as e:
            print(f"Error creating Jira task: {e}")
            return None

# Global Jira client instance
jira_client = JiraClient()