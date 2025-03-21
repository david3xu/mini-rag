import { useState, useCallback } from 'react';
import axios, { AxiosError, AxiosResponse } from 'axios';

/**
 * Generic interface for API state
 */
interface ApiState<T> {
  data: T | null;
  isLoading: boolean;
  error: string | null;
}

/**
 * Interface for useApi hook return value
 */
interface UseApiResult<T, P> {
  state: ApiState<T>;
  executeRequest: (params: P) => Promise<T | null>;
  reset: () => void;
}

/**
 * Generic hook for API calls with TypeScript support
 */
export function useApi<T, P>(
  apiFunction: (params: P) => Promise<AxiosResponse<T>>
): UseApiResult<T, P> {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    isLoading: false,
    error: null
  });

  /**
   * Execute the API request with params
   */
  const executeRequest = useCallback(async (params: P): Promise<T | null> => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const response = await apiFunction(params);
      
      setState({
        data: response.data,
        isLoading: false,
        error: null
      });
      
      return response.data;
    } catch (err) {
      const error = err as Error | AxiosError;
      let errorMessage = 'An unexpected error occurred';
      
      if (axios.isAxiosError(error)) {
        errorMessage = error.response?.data?.detail || error.message;
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      setState({
        data: null,
        isLoading: false,
        error: errorMessage
      });
      
      return null;
    }
  }, [apiFunction]);

  /**
   * Reset the state
   */
  const reset = useCallback(() => {
    setState({
      data: null,
      isLoading: false,
      error: null
    });
  }, []);

  return {
    state,
    executeRequest,
    reset
  };
}