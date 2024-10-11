import flwr as fl

# 戦略の作成
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # すべてのクライアントからサンプリングして学習
    fraction_evaluate=0.3, # 半分のクライアントからサンプリングして検証
    min_fit_clients=3,  # 最小学習クライアント数
    min_evaluate_clients=3,  # 最小検証クライアント数
    min_available_clients=3,  # 最小参加クライアント数
    #initial_parameters=fl.common.ndarrays_to_parameters(params),
    #initial_parameters=None #初期パラメータ
)

# Start Flower server
fl.server.start_server(
  server_address="0.0.0.0:8080",
  config=fl.server.ServerConfig(num_rounds=10),
  strategy=strategy
)