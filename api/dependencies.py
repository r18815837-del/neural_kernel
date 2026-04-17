"""FastAPI dependency injection — singletons for store, manager, etc."""
from __future__ import annotations

import logging
from typing import Any, Dict

from nk_app.core.assistant_manager import AssistantManager
from persistence import SQLiteStore, BaseStore
from persistence.lifecycle import ProjectLifecycle, ArtifactVersionManager, CleanupService
from integration.client_contract import ClientContractService
from persistence.access import OwnershipAccessPolicy
from scripts.run_project_generation import build_assistant_manager, build_llm_assistant_manager

logger = logging.getLogger(__name__)

_manager_instance: AssistantManager | None = None
_store_instance: BaseStore | None = None
_project_store: Dict[str, Any] = {}
_contract_instance: ClientContractService | None = None
_access_policy_instance: OwnershipAccessPolicy | None = None


def get_assistant_manager() -> AssistantManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = build_llm_assistant_manager()
        if _manager_instance is None:
            logger.info("LLM manager unavailable, using standard manager")
            _manager_instance = build_assistant_manager()
        else:
            logger.info("Using LLM-powered manager")
    return _manager_instance


def get_store() -> BaseStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = SQLiteStore(db_path="data/neural_kernel.db")
    return _store_instance


def get_project_store() -> Dict[str, Any]:
    return _project_store


def get_lifecycle() -> ProjectLifecycle:
    return ProjectLifecycle(store=get_store())


def get_version_manager() -> ArtifactVersionManager:
    return ArtifactVersionManager(store=get_store())


def get_cleanup_service() -> CleanupService:
    return CleanupService(store=get_store())


def get_contract_service() -> ClientContractService:
    global _contract_instance
    if _contract_instance is None:
        _contract_instance = ClientContractService(base_url="/api/v1")
    return _contract_instance


def get_access_policy() -> OwnershipAccessPolicy:
    global _access_policy_instance
    if _access_policy_instance is None:
        _access_policy_instance = OwnershipAccessPolicy.from_env()
    return _access_policy_instance
