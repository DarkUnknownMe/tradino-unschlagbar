
class ManualTakeProfitManager:
    def __init__(self, exchange):
        self.exchange = exchange
        self.tp_levels = {}
        
    def set_take_profit(self, symbol, tp_price, position_size):
        self.tp_levels[symbol] = {
            'tp_price': tp_price,
            'size': position_size,
            'active': True
        }
        
    def monitor_take_profits(self):
        for symbol, tp_data in self.tp_levels.items():
            if not tp_data['active']:
                continue
                
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                if current_price >= tp_data['tp_price']:
                    # Execute take profit
                    self.exchange.create_market_order(
                        symbol=symbol,
                        side='sell',
                        amount=tp_data['size']
                    )
                    tp_data['active'] = False
                    
            except Exception as e:
                continue
