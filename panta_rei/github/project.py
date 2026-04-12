"""GitHub Projects V2 management via GraphQL API.

Manages project board status for ALMA scheduling block issues,
supporting automated status updates while preserving manual
reviewer changes.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# GitHub GraphQL endpoint
GH_GRAPHQL_URL = "https://api.github.com/graphql"

# Project board status column names (must match exactly what's in the project)
PROJECT_STATUS_IN_PROGRESS = "In progress"
PROJECT_STATUS_DELIVERED = "Delivered"
PROJECT_STATUS_WEBLOG_QA = "Weblog QA"
PROJECT_STATUS_RECALIBRATION = "Needs re-calibration"
PROJECT_STATUS_REIMAGING = "Needs re-imaging"
PROJECT_STATUS_DONE = "Done"


class GitHubProjectManager:
    """Manages GitHub Projects V2 via GraphQL API."""

    def __init__(self, token: str, org: str, project_number: int):
        self.token = token
        self.org = org
        self.project_number = project_number

        # Cached project metadata
        self._project_id: Optional[str] = None
        self._status_field_id: Optional[str] = None
        self._status_options: Dict[str, str] = {}  # name -> option_id
        self._existing_items: Dict[str, str] = {}  # issue_node_id -> project_item_id
        self._item_statuses: Dict[str, str] = {}  # issue_node_id -> current status name

    def _graphql(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query against the GitHub API."""
        payload: Dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            GH_GRAPHQL_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        if "errors" in result:
            raise Exception(f"GraphQL error: {result['errors']}")

        return result["data"]

    def _cache_items_page(self, items_nodes: list) -> None:
        """Cache project items and their statuses from a page of results."""
        for item in items_nodes:
            if item and item.get("content") and item["content"].get("id"):
                issue_node_id = item["content"]["id"]
                self._existing_items[issue_node_id] = item["id"]

                # Extract current status
                for field_value in item.get("fieldValues", {}).get("nodes", []):
                    if field_value and field_value.get("field", {}).get("name") == "Status":
                        self._item_statuses[issue_node_id] = field_value.get("name", "")
                        break

    def _load_all_items(self) -> None:
        """Paginate through all project items."""
        query = """
        query($projectId: ID!, $cursor: String) {
          node(id: $projectId) {
            ... on ProjectV2 {
              items(first: 100, after: $cursor) {
                pageInfo {
                  hasNextPage
                  endCursor
                }
                nodes {
                  id
                  content {
                    ... on Issue {
                      id
                      number
                    }
                  }
                  fieldValues(first: 10) {
                    nodes {
                      ... on ProjectV2ItemFieldSingleSelectValue {
                        name
                        field {
                          ... on ProjectV2SingleSelectField {
                            name
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        cursor = None
        while True:
            data = self._graphql(query, {
                "projectId": self._project_id,
                "cursor": cursor,
            })
            items = data["node"]["items"]
            self._cache_items_page(items["nodes"])

            page_info = items["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            cursor = page_info["endCursor"]

    def load_project_metadata(self) -> bool:
        """Load project ID, status field ID, status options, and all items."""
        query = """
        query($org: String!, $number: Int!) {
          organization(login: $org) {
            projectV2(number: $number) {
              id
              title
              fields(first: 20) {
                nodes {
                  ... on ProjectV2SingleSelectField {
                    id
                    name
                    options {
                      id
                      name
                    }
                  }
                }
              }
            }
          }
        }
        """

        try:
            data = self._graphql(query, {"org": self.org, "number": self.project_number})
            project = data["organization"]["projectV2"]

            self._project_id = project["id"]
            log.info(f"Loaded project: {project['title']} (ID: {self._project_id})")

            # Find the Status field
            for field_node in project["fields"]["nodes"]:
                if field_node and field_node.get("name") == "Status":
                    self._status_field_id = field_node["id"]
                    for opt in field_node.get("options", []):
                        self._status_options[opt["name"]] = opt["id"]
                    log.info(f"Found Status field with options: {list(self._status_options.keys())}")
                    break

            # Load all items with pagination
            self._load_all_items()

            log.info(f"Found {len(self._existing_items)} existing items in project")

            return True

        except Exception as e:
            log.warning(f"Could not load project metadata: {e}")
            return False

    def add_issue_to_project(self, issue_node_id: str) -> Optional[str]:
        """Add an issue to the project.

        Args:
            issue_node_id: The GraphQL node ID of the issue

        Returns:
            The project item ID, or None if failed
        """
        if not self._project_id:
            log.warning("Project not loaded, cannot add issue")
            return None

        # Check if already in project
        if issue_node_id in self._existing_items:
            return self._existing_items[issue_node_id]

        mutation = """
        mutation($projectId: ID!, $contentId: ID!) {
          addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
            item {
              id
            }
          }
        }
        """

        try:
            data = self._graphql(mutation, {
                "projectId": self._project_id,
                "contentId": issue_node_id,
            })
            item_id = data["addProjectV2ItemById"]["item"]["id"]
            self._existing_items[issue_node_id] = item_id
            return item_id
        except Exception as e:
            log.warning(f"Could not add issue to project: {e}")
            return None

    def set_item_status(self, item_id: str, status_name: str) -> bool:
        """Set the Status field for a project item.

        Args:
            item_id: The project item ID
            status_name: The status option name (e.g., "Weblog QA")

        Returns:
            True if successful
        """
        if not self._status_field_id:
            log.warning("Status field not found, cannot set status")
            return False

        option_id = self._status_options.get(status_name)
        if not option_id:
            log.warning(f"Status option '{status_name}' not found. Available: {list(self._status_options.keys())}")
            return False

        mutation = """
        mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $projectId
            itemId: $itemId
            fieldId: $fieldId
            value: {singleSelectOptionId: $optionId}
          }) {
            projectV2Item {
              id
            }
          }
        }
        """

        try:
            self._graphql(mutation, {
                "projectId": self._project_id,
                "itemId": item_id,
                "fieldId": self._status_field_id,
                "optionId": option_id,
            })
            return True
        except Exception as e:
            log.warning(f"Could not set status to '{status_name}': {e}")
            return False

    def get_issue_node_id(self, owner: str, repo: str, issue_number: int) -> Optional[str]:
        """Get the GraphQL node ID for an issue."""
        query = """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            issue(number: $number) {
              id
            }
          }
        }
        """

        try:
            data = self._graphql(query, {
                "owner": owner,
                "repo": repo,
                "number": issue_number,
            })
            return data["repository"]["issue"]["id"]
        except Exception as e:
            log.warning(f"Could not get issue node ID: {e}")
            return None

    def get_current_status(self, issue_node_id: str) -> Optional[str]:
        """Get the current status of an issue in the project."""
        return self._item_statuses.get(issue_node_id)

    def is_status_manually_advanced(self, current_status: Optional[str]) -> bool:
        """Check if the current status indicates manual progression beyond initial states.

        Returns True if the issue has been moved to a review/work state that shouldn't
        be automatically reverted.
        """
        manual_states = {
            PROJECT_STATUS_WEBLOG_QA,
            PROJECT_STATUS_RECALIBRATION,
            PROJECT_STATUS_REIMAGING,
            PROJECT_STATUS_DONE,
        }
        return current_status in manual_states
