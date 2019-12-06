from gym.envs.registration import register
 
register(id='Interceptor-v2', 
    entry_point='interceptor.envs:InterceptorEnv', 
)
