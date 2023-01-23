from analyze_status import AnalyzeStatus


class ScoringResult2:
    def __init__(self, id: int, analyze_status: int, storage_key_rml: str = None) -> None:
        self.id = id
        self.analyze_status = AnalyzeStatus(analyze_status)
        self.storage_key_rml = storage_key_rml
