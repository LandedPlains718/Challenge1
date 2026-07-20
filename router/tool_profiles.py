#!/usr/bin/env python3
"""Profiles for tools/platforms companies *use* (not build).

"X companies using Y or similar" → search X operators; Y is usually absent from
the corpus, so expand to operator signals and demote Y-class vendors when possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolProfile:
    key: str
    category: str
    aliases: tuple[str, ...] = ()
    # Phrases that describe *users* of this tool class (not the vendor).
    operator_expansions: tuple[str, ...] = ()
    # Short label for answer/evidence framing.
    operator_label: str = "operators that may use this tool"
    # Offering snippets that suggest the company *builds* this class of tool.
    vendor_offering_signals: tuple[str, ...] = ()


_STOREFRONT_EXPANSIONS = (
    "online retail",
    "ecommerce",
    "direct to consumer",
    "online store",
    "webshop",
)
_STOREFRONT_VENDOR = (
    "ecommerce platform development",
    "e-commerce platform development",
    "online shop platform development",
    "shopping cart platform",
    "storefront platform",
    "online shop platform",
)

_PAYMENTS_EXPANSIONS = (
    "payment processing",
    "online payments",
    "digital payments",
    "payment solutions",
)
_PAYMENTS_VENDOR = (
    "payment gateway development",
    "payment platform development",
    "payments infrastructure platform",
    "acquiring platform development",
)

_ERP_EXPANSIONS = (
    "enterprise operations",
    "supply chain operations",
    "warehouse management",
    "ERP systems",
)
_ERP_VENDOR = (
    "erp software development",
    "erp platform development",
    "enterprise resource planning software",
)

_CRM_EXPANSIONS = (
    "customer relationship management",
    "sales operations",
    "crm systems",
)
_CRM_VENDOR = (
    "crm software development",
    "crm platform development",
    "salesforce consulting platform",
)

_EHR_EXPANSIONS = (
    "hospital IT",
    "clinical software",
    "electronic health records",
    "ehr systems",
)
_EHR_VENDOR = (
    "ehr software development",
    "electronic health record platform",
    "clinical information system development",
)

_ANALYTICS_EXPANSIONS = (
    "data analytics",
    "business intelligence",
    "analytics platforms",
)
_ANALYTICS_VENDOR = (
    "analytics platform development",
    "bi platform development",
)


def _storefront(key: str, *aliases: str) -> ToolProfile:
    return ToolProfile(
        key=key,
        category="storefront",
        aliases=aliases,
        operator_expansions=_STOREFRONT_EXPANSIONS,
        operator_label="e-commerce merchants / online retailers",
        vendor_offering_signals=_STOREFRONT_VENDOR,
    )


def _payments(key: str, *aliases: str) -> ToolProfile:
    return ToolProfile(
        key=key,
        category="payments",
        aliases=aliases,
        operator_expansions=_PAYMENTS_EXPANSIONS,
        operator_label="payment / fintech operators",
        vendor_offering_signals=_PAYMENTS_VENDOR,
    )


def _erp(key: str, *aliases: str) -> ToolProfile:
    return ToolProfile(
        key=key,
        category="erp",
        aliases=aliases,
        operator_expansions=_ERP_EXPANSIONS,
        operator_label="enterprise / operations operators",
        vendor_offering_signals=_ERP_VENDOR,
    )


def _crm(key: str, *aliases: str) -> ToolProfile:
    return ToolProfile(
        key=key,
        category="crm",
        aliases=aliases,
        operator_expansions=_CRM_EXPANSIONS,
        operator_label="CRM / sales operators",
        vendor_offering_signals=_CRM_VENDOR,
    )


def _ehr(key: str, *aliases: str) -> ToolProfile:
    return ToolProfile(
        key=key,
        category="ehr",
        aliases=aliases,
        operator_expansions=_EHR_EXPANSIONS,
        operator_label="healthcare providers / clinical operators",
        vendor_offering_signals=_EHR_VENDOR,
    )


PROFILES: tuple[ToolProfile, ...] = (
    _storefront("shopify"),
    _storefront("woocommerce", "woo"),
    _storefront("magento", "adobe commerce"),
    _storefront("bigcommerce"),
    _storefront("squarespace"),
    _storefront("wix"),
    _storefront("prestashop"),
    _payments("stripe"),
    _payments("adyen"),
    _payments("braintree"),
    _payments("paypal"),
    _payments("square"),
    _erp("sap"),
    _erp("oracle"),
    _erp("netsuite", "net suite"),
    _erp("microsoft dynamics", "dynamics"),
    _crm("salesforce"),
    _crm("hubspot"),
    _crm("zendesk"),
    _ehr("epic"),
    _ehr("cerner"),
    _ehr("athenahealth", "athena"),
    ToolProfile(
        key="tableau",
        category="analytics",
        operator_expansions=_ANALYTICS_EXPANSIONS,
        operator_label="analytics / BI operators",
        vendor_offering_signals=_ANALYTICS_VENDOR,
    ),
    ToolProfile(
        key="snowflake",
        category="analytics",
        operator_expansions=("data warehouse", "cloud data platform", "analytics"),
        operator_label="data / analytics operators",
        vendor_offering_signals=("data warehouse platform development",),
    ),
)


def normalize_tool_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def query_mentions_tool_brand(query: str, tool: str) -> bool:
    """True when an embedding/rerank phrase is the brand or brand-led (drop it)."""
    tool_key = normalize_tool_key(tool)
    if len(tool_key) < 3:
        return False
    q_key = normalize_tool_key(query)
    if q_key == tool_key:
        return True
    # "salesforce alternatives", "stripe payments" — brand-led noise
    if q_key.startswith(tool_key):
        return True
    return False


def _build_lookup() -> dict[str, ToolProfile]:
    out: dict[str, ToolProfile] = {}
    for profile in PROFILES:
        out[normalize_tool_key(profile.key)] = profile
        for alias in profile.aliases:
            out[normalize_tool_key(alias)] = profile
    return out


_LOOKUP = _build_lookup()


def lookup_tool(name: str) -> ToolProfile | None:
    if not (name or "").strip():
        return None
    return _LOOKUP.get(normalize_tool_key(name))


def is_known_tool(name: str) -> bool:
    return lookup_tool(name) is not None


def known_tool_keys() -> frozenset[str]:
    """Normalized keys + aliases for brand detection."""
    return frozenset(_LOOKUP.keys())


def operator_expansions_for(tool: str) -> list[str]:
    profile = lookup_tool(tool)
    if not profile:
        return []
    return list(profile.operator_expansions)


def operator_label_for(tool: str, *, primary_theme: str = "") -> str:
    profile = lookup_tool(tool)
    theme = (primary_theme or "").strip()
    brand = (tool or (profile.key if profile else "this tool")).strip()
    if theme:
        return f"{theme} operators that may use {brand}-class tools"
    if profile:
        return f"{profile.operator_label} that may use {brand}-class tools"
    return f"operators that may use {brand}"


def _offerings_blob(row: dict[str, Any]) -> str:
    offs = row.get("core_offerings") or []
    text = " ".join(o for o in offs if isinstance(o, str)).lower()
    text = text.replace("e-commerce", "ecommerce").replace("e commerce", "ecommerce")
    return text


def is_tool_class_vendor(row: dict[str, Any], tool: str) -> bool:
    """True when the company looks like a builder/vendor of the tool's class."""
    profile = lookup_tool(tool)
    if not profile or not profile.vendor_offering_signals:
        return False
    offerings = _offerings_blob(row)
    if not offerings:
        return False
    if not any(sig in offerings for sig in profile.vendor_offering_signals):
        return False
    # Storefront: keep merchants (E-commerce business model) out of vendor bucket.
    if profile.category == "storefront":
        models = " ".join(
            m.lower() for m in (row.get("business_model") or []) if isinstance(m, str)
        )
        if "e-commerce" in models or "ecommerce" in models.replace("-", ""):
            return False
    return True


def is_storefront_tool(tool: str) -> bool:
    profile = lookup_tool(tool)
    return bool(profile and profile.category == "storefront")
