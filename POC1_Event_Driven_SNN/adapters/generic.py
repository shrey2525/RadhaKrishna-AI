from core.event import Event

class GenericAdapter:
    def transform(self, events):
        return list(events)
