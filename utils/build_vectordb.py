import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
import hashlib
import os
import shutil
from typing import List, Tuple

def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "CustomerSupportAIAgent/1.0"})
    return session


def fetch_page(session: requests.Session, url: str) -> Tuple[str, str, str]:
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        return url, title, text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return url, "", ""

# For demo: fetch homepage + 1 subpage each (you can extend this)
DOCS_URLS = [
    "https://docs.atlan.com/get-started/what-is-atlan",
    "https://docs.atlan.com/get-started/how-tos/custom-solutions",
    "https://docs.atlan.com/get-started/how-tos/getting-started-with-the-apis", 
    "https://docs.atlan.com/get-started/how-tos/quick-start-for-admins",
    "https://docs.atlan.com/get-started/how-tos/quick-start-for-contributors",
    "https://docs.atlan.com/get-started/how-tos/quick-start-for-data-consumers",
    "https://docs.atlan.com/product/administration/labs/how-tos/allow-guests-to-request-updates",
    "https://docs.atlan.com/product/administration/labs/how-tos/allow-members-to-view-reports",
    "https://docs.atlan.com/product/administration/labs/how-tos/enable-discovery-of-process-assets",
    "https://docs.atlan.com/product/administration/labs/how-tos/enable-sample-data-download",
    "https://docs.atlan.com/product/administration/labs/how-tos/restrict-asset-visibility",
    "https://docs.atlan.com/product/capabilities/data-products/how-tos/create-data-products",
    "https://docs.atlan.com/product/capabilities/governance/access-control/how-tos/create-a-persona",
    "https://docs.atlan.com/product/capabilities/lineage/how-tos/view-lineage",
    "https://docs.atlan.com/product/capabilities/lineage/troubleshooting/troubleshooting-lineage",
    "https://docs.atlan.com/product/integrations/identity-management/sso/troubleshooting/troubleshooting-sso",
    "https://docs.atlan.com/product/integrations/identity-management/sso/troubleshooting/troubleshooting-connector-specific-sso-authentication",
    "https://docs.atlan.com/product/integrations/identity-management/sso/faq/pingfederate-404-error",
    "https://docs.atlan.com/product/integrations/identity-management/sso/faq/okta-first-time-login-error",
    "https://docs.atlan.com/product/integrations/identity-management/sso/faq/google-dashboard-login-error",
    "https://docs.atlan.com/product/integrations/identity-management/sso/troubleshooting/microsoft-defender-sso-error",
    "https://docs.atlan.com/apps/connectors/data-warehouses/snowflake/how-tos/set-up-snowflake"
]

DEV_URLS = [
    "https://developer.atlan.com/",
    "https://developer.atlan.com/models/api",
    "https://developer.atlan.com/models/entities/apifield",
    "https://developer.atlan.com/models/entities/apiobject",
    "https://developer.atlan.com/models/entities/apipath",
    "https://developer.atlan.com/models/entities/apiquery",
    "https://developer.atlan.com/models/entities/apispec",
    "https://developer.atlan.com/models/enums/apiqueryparamtypeenum",
    "https://developer.atlan.com/news/2023/08/14/atlan-java-sdk-v100/",
    "https://developer.atlan.com/sdks/cli",
    "https://developer.atlan.com/sdks/java",
    "https://developer.atlan.com/sdks/kotlin",
    "https://developer.atlan.com/sdks/python",
    "https://developer.atlan.com/sdks/scala",
    "https://developer.atlan.com/snippets/datacontract/manage-via-sdks/",
    "https://developer.atlan.com/toolkits/typedef/bind-sdks/"
    ]

def chunk_texts(texts: List[Tuple[str, str, str]], chunk_size: int = 1000, overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", ".", " "], chunk_size=chunk_size, chunk_overlap=overlap)
    all_chunks = []
    all_metas = []
    seen_hashes = set()
    access_visibility = os.environ.get("ACCESS_VISIBILITY", "public")
    access_dept = os.environ.get("ACCESS_DEPT", "support")
    for url, title, text in texts:
        if not text:
            continue
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            content_hash = hashlib.sha1((url + "::" + chunk).encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            all_chunks.append(chunk)
            all_metas.append({"source": url, "title": title, "chunk_index": idx, "content_hash": content_hash, "access_visibility": access_visibility, "access_dept": access_dept})
    return all_chunks, all_metas

session = make_session()
targets = DOCS_URLS + DEV_URLS
fetched: List[Tuple[str, str, str]] = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_to_url = {executor.submit(fetch_page, session, url): url for url in targets}
    for future in concurrent.futures.as_completed(future_to_url):
        fetched.append(future.result())

texts, metadatas = chunk_texts(fetched)
print(f"✅ Total chunks: {len(texts)}")

persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
wipe = os.environ.get("WIPE_CHROMA", "false").lower() == "true"
if wipe and os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(collection_name="db_docs", embedding_function=embedding, persist_directory=persist_dir)

ids = [m["content_hash"] for m in metadatas]
batch_size = 200
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_metas = metadatas[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]
    try:
        vectordb.add_texts(texts=batch_texts, metadatas=batch_metas, ids=batch_ids)
    except Exception as e:
        print(f"Failed to add batch {i}-{i+len(batch_texts)}: {e}")

print(f"🎉 Knowledge base built and saved to: {persist_dir}")
