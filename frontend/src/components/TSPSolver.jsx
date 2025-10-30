import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { Badge } from './ui/badge';
import { Loader2, Play, RefreshCw, CheckCircle, AlertCircle, MapPin } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function TSPSolver() {
  const [numCities, setNumCities] = useState(3);
  const [pLayers, setPLayers] = useState(2);
  const [maxIter, setMaxIter] = useState(50);
  const [distanceMatrix, setDistanceMatrix] = useState('[[0,10,15],[10,0,20],[15,20,0]]');
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const generateRandom = async () => {
    setGenerating(true);
    setError(null);
    try {
      const response = await axios.post(`${API}/generate/random`, {
        problem_type: 'tsp',
        size: numCities,
        additional_params: {
          max_distance: 50
        }
      });
      
      setDistanceMatrix(JSON.stringify(response.data.distance_matrix));
    } catch (err) {
      setError(err.response?.data?.detail || 'Error generating TSP instance');
    } finally {
      setGenerating(false);
    }
  };

  const solveProblem = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const matrix = JSON.parse(distanceMatrix);
      
      // Validate matrix
      if (!Array.isArray(matrix) || matrix.length === 0) {
        throw new Error('Invalid distance matrix format');
      }
      
      if (matrix.length > 4) {
        throw new Error('Maximum 4 cities allowed for quantum simulation');
      }

      const response = await axios.post(`${API}/tsp/solve`, {
        distance_matrix: matrix,
        p_layers: pLayers,
        max_iter: maxIter,
        method: 'COBYLA'
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Error solving problem');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Panel */}
      <Card data-testid="tsp-input-panel">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MapPin className="w-5 h-5" />
            Traveling Salesman Problem (TSP)
          </CardTitle>
          <CardDescription>
            Find the shortest route visiting all cities exactly once
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <AlertDescription className="text-sm">
              <strong>Limit:</strong> Maximum 4 cities for quantum simulation feasibility
            </AlertDescription>
          </Alert>

          <div className="space-y-2">
            <Label htmlFor="num-cities">Number of Cities</Label>
            <Input
              id="num-cities"
              type="number"
              min="2"
              max="4"
              value={numCities}
              onChange={(e) => setNumCities(Number(e.target.value))}
              data-testid="num-cities-input"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="distance-matrix">Distance Matrix (JSON format)</Label>
            <textarea
              id="distance-matrix"
              value={distanceMatrix}
              onChange={(e) => setDistanceMatrix(e.target.value)}
              placeholder='[[0,10,15],[10,0,20],[15,20,0]]'
              className="w-full h-32 px-3 py-2 border rounded-md font-mono text-sm"
              data-testid="distance-matrix-input"
            />
            <p className="text-xs text-gray-500">
              Example for 3 cities: [[0,10,15],[10,0,20],[15,20,0]]
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="p-layers">QAOA Layers (p)</Label>
              <Select
                id="p-layers"
                value={pLayers}
                onChange={(e) => setPLayers(Number(e.target.value))}
                data-testid="p-layers-select"
              >
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="max-iter">Max Iterations</Label>
              <Select
                id="max-iter"
                value={maxIter}
                onChange={(e) => setMaxIter(Number(e.target.value))}
                data-testid="max-iter-select"
              >
                <option value="30">30</option>
                <option value="50">50</option>
                <option value="100">100</option>
              </Select>
            </div>
          </div>

          <div className="flex gap-2">
            <Button
              onClick={solveProblem}
              disabled={loading || generating}
              className="flex-1"
              data-testid="solve-button"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Solving...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Solve with QAOA
                </>
              )}
            </Button>

            <Button
              onClick={generateRandom}
              disabled={loading || generating}
              variant="outline"
              data-testid="generate-button"
            >
              {generating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
            </Button>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="w-4 h-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Results Panel */}
      <Card data-testid="tsp-results-panel">
        <CardHeader>
          <CardTitle>Results</CardTitle>
          <CardDescription>QAOA optimization results and tour analysis</CardDescription>
        </CardHeader>
        <CardContent>
          {!result && !loading && (
            <div className="text-center py-12 text-gray-400">
              <MapPin className="w-16 h-16 mx-auto mb-4" />
              <p>Configure TSP and run QAOA to see results</p>
            </div>
          )}

          {loading && (
            <div className="text-center py-12">
              <Loader2 className="w-12 h-12 animate-spin mx-auto text-blue-600" />
              <p className="mt-4 text-gray-600">Running quantum optimization...</p>
            </div>
          )}

          {result && (
            <div className="space-y-6">
              {/* Status Badge */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {result.is_valid ? (
                    <>
                      <CheckCircle className="w-5 h-5 text-green-600" />
                      <Badge className="bg-green-100 text-green-800">Valid Tour</Badge>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-5 h-5 text-orange-600" />
                      <Badge className="bg-orange-100 text-orange-800">Approximate Solution</Badge>
                    </>
                  )}
                </div>
                <span className="text-sm text-gray-600">
                  {result.iterations} iterations
                </span>
              </div>

              {/* Key Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Tour Cost</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {result.tour_cost.toFixed(2)}
                  </p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Approximation Ratio</p>
                  <p className="text-3xl font-bold text-green-600">
                    {(result.classical_comparison.qaoa_approximation_ratio * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Tour Details */}
              <div>
                <Label className="mb-2 block">Optimal Tour</Label>
                <div className="bg-gray-50 p-4 rounded">
                  <p className="font-mono text-lg text-center text-blue-700 font-semibold">
                    {result.tour_string}
                  </p>
                  <p className="text-sm text-gray-600 text-center mt-2">
                    Total Distance: <span className="font-semibold">{result.tour_cost.toFixed(2)}</span>
                  </p>
                </div>
              </div>

              {/* Performance Comparison */}
              <div>
                <Label className="mb-2 block">Performance Comparison</Label>
                <div className="space-y-2 bg-gray-50 p-4 rounded">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">QAOA Result:</span>
                    <Badge variant="default">{result.tour_cost.toFixed(2)}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Classical (Greedy):</span>
                    <Badge variant="secondary">
                      {result.classical_comparison.classical_cost.toFixed(2)}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Optimal:</span>
                    <Badge className="bg-green-100 text-green-800">
                      {result.classical_comparison.optimal_cost.toFixed(2)}
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Convergence Analysis */}
              {result.convergence_analysis && (
                <div>
                  <Label className="mb-2 block">Convergence Analysis</Label>
                  <div className="bg-gray-50 p-4 rounded space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Initial Cost:</span>
                      <span className="font-mono">
                        {result.convergence_analysis.initial_cost?.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Final Cost:</span>
                      <span className="font-mono">
                        {result.convergence_analysis.final_cost?.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Improvement:</span>
                      <Badge className="bg-green-100 text-green-800">
                        {result.convergence_analysis.improvement?.toFixed(2)}%
                      </Badge>
                    </div>
                  </div>
                </div>
              )}

              {/* Problem Info */}
              <div className="text-xs text-gray-500 space-y-1">
                <div className="flex justify-between">
                  <span>Number of Cities:</span>
                  <span>{result.problem_info.num_cities}</span>
                </div>
                <div className="flex justify-between">
                  <span>Qubits Used:</span>
                  <span>{result.problem_info.num_qubits}</span>
                </div>
                <div className="flex justify-between">
                  <span>QAOA Depth:</span>
                  <span>p = {result.problem_info.p_layers}</span>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
