import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Weekend Planner MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: generate_weekend_plan ---
# System prompt (configured on PuchAI side):
# "You are an expert weekend planner who creates personalized, engaging weekend plans based on user preferences for staying in or going out, budget, interests, and location."
@mcp.tool(description="Generates a weekend plan based on user preferences")
async def generate_weekend_plan(
    user_preference: Annotated[str, Field(description="Stay in or go out")],
    interests: Annotated[str | None, Field(description="User interests, e.g., movies, parks, baking")] = None,
    location: Annotated[str | None, Field(description="User city or location")] = None,
    budget: Annotated[str | None, Field(description="Budget level: low, medium, high")] = None,
) -> str:
    return (
        f"User Preferences:\n"
        f"- Stay in or go out: {user_preference}\n"
        f"- Interests: {interests or 'general activities'}\n"
        f"- Location: {location or 'unspecified'}\n"
        f"- Budget: {budget or 'medium'}"
    )

# --- Tool: surprise_me ---
# System prompt:
# "You are a creative weekend planner who suggests surprising and fun activities for the weekend."
@mcp.tool(description="Returns a fun surprise weekend plan")
async def surprise_me() -> str:
    return "User requests a surprise weekend plan."

# --- Tool: format_share_text ---
# System prompt:
# "You are an expert at formatting plans into clear, friendly text ideal for sharing with friends or family."
@mcp.tool(description="Formats a weekend plan text for easy sharing")
async def format_share_text(
    plan_text: Annotated[str, Field(description="Raw plan text")]
) -> str:
    return plan_text

# --- Tool: raksha_bandhan_special ---
# System prompt:
# "You are a cultural events expert who suggests fun and memorable Raksha Bandhan sibling activities."
@mcp.tool(description="Special Raksha Bandhan sibling activities")
async def raksha_bandhan_special() -> str:
    return "User requests special Raksha Bandhan sibling activities."

# --- Tool: weather_prompt ---
# System prompt:
# "You are a helpful assistant who politely asks the user about the current weather to tailor suggestions."
@mcp.tool(description="Ask user about local weather to tailor suggestions")
async def weather_prompt() -> str:
    return "Please ask the user about the current weather (e.g., sunny, rainy, chilly) to better tailor weekend plans."

# --- Tool: easter_eggs ---
# System prompt:
# "You are a witty and empathetic weekend planner who responds creatively to special user phrases such as financial constraints."
@mcp.tool(description="Responds to special phrases like 'I have no money'")
async def easter_eggs(
    trigger_phrase: Annotated[str, Field(description="User's special phrase")]
) -> str:
    return f"User says: {trigger_phrase}"

# --- Tool: location_search ---
# System prompt:
# "You are a location-aware weekend guide who finds fun activities or places based on the user's location and interest queries."
@mcp.tool(description="Searches for fun things to do based on location and interest")
async def location_search(
    location_query: Annotated[str, Field(description="Location name, e.g., city or neighborhood")],
    activity_query: Annotated[str | None, Field(description="Type of activity user is interested in")] = None,
) -> str:
    return (
        f"Location Query: {location_query}\n"
        f"Activity Query: {activity_query or 'any'}"
    )


# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
