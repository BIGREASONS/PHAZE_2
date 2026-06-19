import os

replacements = {
    "ASTraM Command Center": "GridSight AI Command Center",
    "ASTraM road-closure prediction": "GridSight AI road-closure prediction",
    "ASTraM: Shift-Aware": "GridSight AI: Shift-Aware",
    "ASTraM Traffic Incident Command Center": "GridSight AI Traffic Incident Command Center",
    "ASTraM Intelligence System": "GridSight AI Intelligence System",
    "ASTraM 7-Model": "GridSight AI 7-Model",
    "ASTraM final model": "GridSight AI final model",
    "ASTraM road-closure stack": "GridSight AI road-closure stack",
    "ASTraM — Shift-Aware": "GridSight AI — Shift-Aware",
    "ASTraM — Reproducibility": "GridSight AI — Reproducibility",
    "ASTraM — Production": "GridSight AI — Production",
    "ASTraM — Streamlit": "GridSight AI — Streamlit",
    "ASTraM — Backend": "GridSight AI — Backend",
    "ASTraM — Data Service": "GridSight AI — Data Service",
    "ASTraM — Metrics": "GridSight AI — Metrics",
    "ASTraM — Model Adapter": "GridSight AI — Model Adapter",
    "ASTraM — Application": "GridSight AI — Application",
    "ASTraM — FastAPI": "GridSight AI — FastAPI",
    "ASTraM — Lightweight": "GridSight AI — Lightweight",
    "ASTraMPDF": "GridSightPDF"
}

files_to_check = [
    "docs/TECHNICAL_REPORT.md",
    "docs/PORTFOLIO_SHOWCASE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/DEPLOYMENT_CHECKLIST.md",
    "command_center/backend/main.py",
    "command_center/backend/services/model_adapter.py",
    "command_center/backend/services/pdf_generator.py",
    "command_center/backend/services/auth.py",
    "command_center/backend/services/data_service.py",
    "command_center/backend/services/metrics_registry.py",
    "command_center/backend/config.py",
    "command_center/backend/__init__.py",
    "command_center/frontend/app.py",
    "command_center/frontend/__init__.py",
    "command_center/frontend/components/ui.py",
    "command_center/frontend/components/maps.py",
    "command_center/frontend/components/theme.py",
    "command_center/__init__.py",
    "command_center/README.md",
    "final_validation/artifacts/metadata.json"
]

for fp in files_to_check:
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        
        changed = False
        for k, v in replacements.items():
            if k in content:
                content = content.replace(k, v)
                changed = True
        
        if changed:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated {fp}")

print("Done.")
