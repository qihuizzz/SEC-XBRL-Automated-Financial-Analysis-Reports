from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import requests


@dataclass
class SecClientConfig:
    user_agent: str
    data_base_url: str = "https://data.sec.gov"
    sec_base_url: str = "https://www.sec.gov"
    cache_dir: str = ".cache"
    timeout_seconds: int = 30
    max_retries: int = 3
    min_interval_seconds: float = 0.2  # simple rate limit


class SecClient:
    """
    Minimal SEC client

    Responsibilities
    - GET JSON with required headers
    - basic retry with backoff
    - basic rate limiting
    - local JSON cache
    """

    def __init__(self, config: SecClientConfig):
        self.cfg = config
        self._session = requests.Session()
        self._last_request_ts = 0.0

        self._cache_path = Path(self.cfg.cache_dir)
        self._cache_path.mkdir(parents=True, exist_ok=True)

        # Do not set Host manually
        self._session.headers.update(
            {
                "User-Agent": self.cfg.user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def _sleep_for_rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_ts
        if elapsed < self.cfg.min_interval_seconds:
            time.sleep(self.cfg.min_interval_seconds - elapsed)
        self._last_request_ts = time.time()

    @staticmethod
    def _make_cache_filename(key: str) -> str:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return f"{h}.json"

    def _cache_key_to_path(self, cache_key: str) -> Path:
        return self._cache_path / self._make_cache_filename(cache_key)

    def get_json(self, url: str, cache_key: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Fetch JSON with cache + retry

        url can be
        - full url starting with https
        - path under data.sec.gov if it starts with /
        """
        if url.startswith("http"):
            full_url = url
        elif url.startswith("/"):
            full_url = f"{self.cfg.data_base_url}{url}"
        else:
            # fallback
            full_url = f"{self.cfg.data_base_url}/{url}"

        key = cache_key or full_url
        cache_file = self._cache_key_to_path(key)

        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        last_err: Optional[Exception] = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                self._sleep_for_rate_limit()
                resp = self._session.get(full_url, timeout=self.cfg.timeout_seconds)

                # Too many requests
                if resp.status_code == 429:
                    time.sleep(0.8 * attempt)
                    continue

                # Some SEC endpoints may return 403 if User-Agent is missing or unacceptable
                if resp.status_code == 403:
                    raise RuntimeError(
                        "SEC returned 403. Check User-Agent string and slow down requests."
                    )

                resp.raise_for_status()

                data = resp.json()
                cache_file.write_text(json.dumps(data), encoding="utf-8")
                return data

            except Exception as e:
                last_err = e
                time.sleep(0.6 * attempt)

        raise RuntimeError(f"SEC request failed after retries: {full_url}") from last_err

    @staticmethod
    def cik_pad10(cik: int | str) -> str:
        s = str(cik).strip()
        return s.zfill(10)

    def ticker_to_cik(self, ticker: str, use_cache: bool = True) -> str:
        """
        Map ticker to 10 digit CIK using company_tickers.json on sec.gov
        """
        t = ticker.strip().upper()

        mapping = self.get_json(
            f"{self.cfg.sec_base_url}/files/company_tickers.json",
            cache_key="sec_company_tickers_json",
            use_cache=use_cache,
        )

        # mapping format: dict of numeric keys -> {cik_str, ticker, title}
        for _, row in mapping.items():
            if str(row.get("ticker", "")).upper() == t:
                return self.cik_pad10(row["cik_str"])

        raise ValueError(f"Ticker not found in SEC mapping: {t}")

    def get_submissions(self, cik10: str, use_cache: bool = True) -> Dict[str, Any]:
        cik10 = self.cik_pad10(cik10)
        return self.get_json(
            f"/submissions/CIK{cik10}.json",
            cache_key=f"submissions_{cik10}",
            use_cache=use_cache,
        )

    def get_companyfacts(self, cik10: str, use_cache: bool = True) -> Dict[str, Any]:
        cik10 = self.cik_pad10(cik10)
        return self.get_json(
            f"/api/xbrl/companyfacts/CIK{cik10}.json",
            cache_key=f"companyfacts_{cik10}",
            use_cache=use_cache,
        )


def make_default_client() -> SecClient:
    # Please keep a real contact in User-Agent
    user_agent = "Qihui Zhang qihuiz@uchicago.edu"
    cfg = SecClientConfig(user_agent=user_agent)
    return SecClient(cfg)