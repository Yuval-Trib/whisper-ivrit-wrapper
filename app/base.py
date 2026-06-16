from abc import ABC, abstractmethod


class BaseHandler(ABC):

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def handle(self, job_input: dict) -> dict: ...

    def dispatch(self, job_input: dict) -> dict:
        if job_input.get("warmup"):
            return {"status": "warm"}
        return self.handle(job_input)
