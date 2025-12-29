import requests

def fetch_figma_context_from_mcp(figma_url: str, access_token: str, mcp_url: str = "http://localhost:8080/context/figma"):
    """
    Fetch Figma design context from an MCP server.
    Args:
        figma_url: The full Figma file URL
        access_token: Figma personal access token
        mcp_url: MCP server endpoint for Figma context
    Returns:
        dict: Figma context as returned by MCP, or None on error
    """
    payload = {"figma_url": figma_url, "access_token": access_token}
    try:
        resp = requests.post(mcp_url, json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching Figma context: {e}")
        return None
