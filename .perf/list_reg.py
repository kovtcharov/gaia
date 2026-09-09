from gaia_agent.agent import GaiaAgent, GaiaAgentConfig
a = GaiaAgent(config=GaiaAgentConfig(silent_mode=True))
names = sorted(a._tools_registry)
print(len(names))
print("\n".join(names))
