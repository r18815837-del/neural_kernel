from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .artifact_spec import ArtifactSpec
from .client_brief import ClientBrief
from .client_request import ClientRequest
from .feature_spec import FeatureSpec
from .project_spec import ProjectSpec
from .tech_stack_spec import TechStackSpec


@dataclass
class RequirementParser:
    default_project_type: str = "application"
    default_output_format: str = "zip"

    def parse_brief(self, request: ClientRequest) -> ClientBrief:
        raw_text = request.raw_text.strip()
        if not raw_text:
            raise ValueError("ClientRequest.raw_text cannot be empty")

        title = self._infer_title(raw_text)
        summary = raw_text

        requested_features = self._infer_features(raw_text)
        constraints = self._infer_constraints(raw_text)
        target_users = self._infer_target_users(raw_text)
        delivery_format = self._infer_delivery_format(raw_text)

        return ClientBrief(
            title=title,
            summary=summary,
            goals=[summary],
            constraints=constraints,
            requested_features=requested_features,
            target_users=target_users,
            delivery_format=delivery_format or self.default_output_format,
            metadata={
                "source": "requirement_parser",
                "request_user_id": request.user_id,
                "request_session_id": request.session_id,
            },
        )

    def brief_to_project_spec(self, brief: ClientBrief) -> ProjectSpec:
        features = [
            FeatureSpec(
                name=self._normalize_feature_name(feature),
                description=self._describe_feature(feature),
                priority="high",
                required=True,
            )
            for feature in brief.requested_features
        ]

        tech_stack = self._infer_tech_stack(brief.summary)
        project_name = self._slugify(brief.title)

        return ProjectSpec(
            project_name=project_name,
            summary=brief.summary,
            project_type=self.default_project_type,
            features=features,
            tech_stack=tech_stack,
            output_format=brief.delivery_format or self.default_output_format,
            target_platforms=["web"],
            constraints=brief.constraints,
            metadata={
                "source": "requirement_parser",
                "target_users": brief.target_users,
            },
        )

    def _describe_feature(self, feature: str) -> str:
        descriptions = {
            "admin_panel": "Admin panel for management and control",
            "client_database": "Client database management",
            "export": "Export data to external formats",
            "api": "Backend API for application logic and integrations",
            "frontend": "Frontend user interface",
            "roles": "Role-based access control",
            "auth": "Authentication and user access",
            "telegram_bot": "Telegram bot integration",
            "payments": "Payment processing support",
            "core_application": "Core application functionality",
        }

        return descriptions.get(feature, feature.replace("_", " ").capitalize())

    def project_to_artifact_spec(self, project_spec: ProjectSpec) -> ArtifactSpec:
        return ArtifactSpec(
            artifact_name=f"{project_spec.project_name}_artifact",
            artifact_type="project_bundle",
            include_readme=True,
            include_tests=True,
            include_env_example=True,
            packaging=project_spec.output_format,
            metadata={
                "source_project": project_spec.project_name,
            },
        )

    def parse(
        self,
        request: ClientRequest,
    ) -> tuple[ClientBrief, ProjectSpec, ArtifactSpec]:
        brief = self.parse_brief(request)
        project_spec = self.brief_to_project_spec(brief)
        artifact_spec = self.project_to_artifact_spec(project_spec)
        return brief, project_spec, artifact_spec

    def _infer_title(self, raw_text: str) -> str:
        text = raw_text.strip()

        lowered = text.lower()

        if "crm" in lowered:
            return "CRM Project"
        if "telegram" in lowered or "телеграм" in lowered:
            return "Telegram Bot Project"
        if "landing" in lowered or "лендинг" in lowered:
            return "Landing Page Project"
        if "backend" in lowered or "api" in lowered:
            return "Backend Project"

        words = text.split()
        short_title = " ".join(words[:4]).strip()

        if not short_title:
            return "Project"

        return short_title

    def _infer_features(self, raw_text: str) -> List[str]:
        text = raw_text.lower()
        features: List[str] = []

        keyword_map = {
            "auth": ["auth", "авториза", "login", "регистрац"],
            "admin_panel": ["админ", "admin", "dashboard", "панел"],
            "client_database": ["клиент", "crm", "база"],
            "roles": ["роль", "roles", "permissions", "доступ"],
            "export": ["export", "экспорт", "excel", "csv"],
            "api": ["api", "backend", "fastapi", "rest"],
            "frontend": ["frontend", "react", "ui", "интерфейс"],
            "telegram_bot": ["telegram", "телеграм", "bot", "бот"],
            "payments": ["payment", "оплат", "stripe", "касса"],
        }

        for feature_name, keywords in keyword_map.items():
            if any(keyword in text for keyword in keywords):
                features.append(feature_name)

        if not features:
            features.append("core_application")

        return features

    def _infer_constraints(self, raw_text: str) -> List[str]:
        text = raw_text.lower()
        constraints: List[str] = []

        if "zip" in text or "архив" in text:
            constraints.append("deliver_as_zip")

        if "flutter" in text:
            constraints.append("must_be_flutter_compatible")

        if "docker" in text:
            constraints.append("include_docker_support")

        return constraints

    def _infer_target_users(self, raw_text: str) -> List[str]:
        text = raw_text.lower()
        users: List[str] = []

        if "админ" in text or "admin" in text:
            users.append("admin")

        if "клиент" in text or "client" in text:
            users.append("client")

        if not users:
            users.append("end_user")

        return users

    def _infer_delivery_format(self, raw_text: str) -> Optional[str]:
        text = raw_text.lower()

        if "zip" in text or "архив" in text:
            return "zip"
        if "repo" in text or "repository" in text or "git" in text:
            return "repo"
        if "folder" in text or "папк" in text:
            return "folder"

        return None

    def _infer_tech_stack(self, raw_text: str) -> TechStackSpec:
        text = raw_text.lower()

        backend = None
        frontend = None
        database = None
        mobile = None
        deployment = None
        testing: List[str] = []
        integrations: List[str] = []

        if "fastapi" in text:
            backend = "FastAPI"
        elif "django" in text:
            backend = "Django"
        elif "backend" in text or "api" in text:
            backend = "FastAPI"

        if "react" in text:
            frontend = "React"
        elif "frontend" in text or "ui" in text:
            frontend = "React"

        if "postgres" in text or "postgresql" in text:
            database = "PostgreSQL"
        elif "mysql" in text:
            database = "MySQL"
        elif "sqlite" in text:
            database = "SQLite"

        if "flutter" in text:
            mobile = "Flutter"

        if "docker" in text:
            deployment = "Docker"

        if "pytest" in text or "test" in text or "тест" in text:
            testing.append("pytest")

        if "telegram" in text or "телеграм" in text:
            integrations.append("Telegram Bot API")

        # fallback for CRM/business apps
        if "crm" in text or "салон" in text or "клиент" in text:
            backend = backend or "FastAPI"
            frontend = frontend or "React"
            database = database or "PostgreSQL"
            testing = testing or ["pytest"]

        return TechStackSpec(
            backend=backend,
            frontend=frontend,
            database=database,
            mobile=mobile,
            deployment=deployment,
            testing=testing,
            integrations=integrations,
        )

    def _normalize_feature_name(self, feature: str) -> str:
        return self._slugify(feature)

    def _slugify(self, value: str) -> str:
        value = value.strip().lower()

        translit_map = {
            "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
            "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
            "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
            "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
            "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch",
            "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
            "э": "e", "ю": "yu", "я": "ya",
        }

        converted = []
        for ch in value:
            if "a" <= ch <= "z" or "0" <= ch <= "9":
                converted.append(ch)
            elif ch in translit_map:
                converted.append(translit_map[ch])
            elif ch in {" ", "-", ".", ","}:
                converted.append("_")
            else:
                converted.append("_")

        normalized = "".join(converted)

        while "__" in normalized:
            normalized = normalized.replace("__", "_")

        normalized = normalized.strip("_")

        if not normalized:
            normalized = "project"

        # ограничиваем длину имени папки
        normalized = normalized[:40].rstrip("_")

        return normalized or "project"