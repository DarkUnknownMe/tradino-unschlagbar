
class ManualStopLossManager:
    def __init__(self, exchange):
        self.exchange = exchange
        self.stop_levels = {}
        
    def set_stop_loss(self, symbol, stop_price, position_size):
        self.stop_levels[symbol] = {
            'stop_price': stop_price,
            'size': position_size,
            'active': True
        }
        
    def monitor_stops(self):
        for symbol, stop_data in self.stop_levels.items():
            if not stop_data['active']:
                continue
                
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                if current_price <= stop_data['stop_price']:
                    # Execute stop loss
                    self.exchange.create_market_order(
                        symbol=symbol,
                        side='sell',
                        amount=stop_data['size']
                    )
                    stop_data['active'] = False
                    
            except Exception as e:
                continue
