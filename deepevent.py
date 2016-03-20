import requests
import time
import base64
import io

class Event(object):
    def __init__(self, graph, name, etype, data):
        self.obj = {
            "graph": graph,
            "name": name,
            "type": etype,
            "value": data,
            "ts": int(time.time()),
        }

    def data(self):
        return self.obj

class Events(Event):
    def __init__(self, *evts):
        self.events = evts

    def data(self):
        return {"events": list(map(lambda x: x.data(), self.events))}

class EventCreator(object):
    def __init__(self, graph, name, gtype):
        if name is None:
            name = graph
        self.graph = graph
        self.name = name
        self.type = gtype

    def new(self, data):
        return Event(self.graph, self.name, self.type, data)

class StepScalarEventCreator(EventCreator):
    def __init__(self, graph, name=None):
        super().__init__(graph, name, "step_scalar")

    def new(self, step, val):
        return super().new({"step": int(step), "value": float(val)})

class StepScalarEMAEventCreator(StepScalarEventCreator):
    def __init__(self, graph, name=None, alpha=None, n=10):
        super().__init__(graph, name)
        self._val = None
        if alpha is None:
            alpha = 1.0 / (n + 1)
        self._alpha = alpha

    def new(self, step, val):
        val = float(val)
        if self._val is None:
            self._val = val
        else:
            self._val = self._alpha * val + (1 - self._alpha) * self._val
        return super().new(step, self._val)

class StepImageEventCreator(EventCreator):
    def __init__(self, graph, name=None):
        super().__init__(graph, name, "step_image")

    def new(self, step, img):
        out = io.BytesIO()
        img.save(out, "PNG")
        return super().new({"step": int(step), "value": base64.standard_b64encode(out.getvalue()).decode("utf-8")})

class DeepEventBase(object):
    def __init__(self):
        pass

    def send(self, evt):
        return True

class DeepEventDummy(DeepEventBase):
    pass

class DeepEvent(DeepEventBase):
    def __init__(self, url, name):
        self.url = url + "/data/" + name + "/add_event"

    def send(self, evt):
        r = requests.post(self.url, json=evt.data())
        if r.status_code != 200:
            print("Error in POST request:", str(r))
        return r.status_code == 200
