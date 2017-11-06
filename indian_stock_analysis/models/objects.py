import collections

class Stock():
    symbol = ""
    indicatorVal = 0
    wap = 0
    max_quantity = 0


class Indicator():
    name = ''
    period = 0
    fastperiod = 0
    slowperiod = 0
    timeperiod1 = 7
    timeperiod2 = 14
    timeperiod3 = 28


class DayTrades():
    date = 0
    buys = []
    sells = []

    # without init the object will have previous values whenever initialized
    def __init__(self):
        super().__init__()
        self.date = 0
        self.buys = []
        self.sells = []


class ReqParam():
    condition = 0
    rollingPeriod = None
    indicatorPeriod = None
    fastPeriod = None
    slowPeriod = None
    adx_min = 0
    isBuy = None

    def __init__(self):
        super().__init__()
        self.condition = 0
        self.rollingPeriod = None
        self.indicatorPeriod = None
        self.fastPeriod = None
        self.slowPeriod = None
        self.adx_min = None
        self.isBuy = None

    # repr function calling return str(self.__dict__)
    # then return only keys that are not None
    def __repr__(self):
        return str({k: v for k, v in collections.OrderedDict(self.__dict__).items() if v is not None})
