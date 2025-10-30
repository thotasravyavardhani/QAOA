import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { Badge } from './ui/badge';
import { Loader2, Play, RefreshCw, CheckCircle, AlertCircle, MapPin, TrendingUp, BarChart3 } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import QuantumCircuitVisual from './QuantumCircuitVisual';
import { TSPFormula } from './MathFormula';

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
        additional_params: { max_distance: 50 }
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
      if (!Array.isArray(matrix) || matrix.length === 0) throw new Error('Invalid distance matrix format');
      if (matrix.length > 4) throw new Error('Maximum 4 cities allowed');

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

  const getConvergenceData = () => {
    if (!result?.convergence_analysis) return [];
    const { initial_cost, final_cost, iterations } = result.convergence_analysis;
    return Array.from({ length: Math.min(iterations, 20) }, (_, i) => ({
      iteration: i,
      cost: initial_cost + (final_cost - initial_cost) * (i / iterations)
    }));
  };

  return (
    <div className="space-y-6">
      <TSPFormula />

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <Card className="lg:col-span-2 shadow-lg border-2 border-green-100" data-testid="tsp-input-panel">
          <CardHeader className="bg-gradient-to-r from-green-600 to-teal-600 text-white">
            <CardTitle className="flex items-center gap-2">
              <MapPin className="w-5 h-5" />
              TSP Configuration
            </CardTitle>
            <CardDescription className="text-green-100">Find shortest tour visiting all cities</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-6">
            <Alert className="bg-amber-50 border-amber-300">
              <AlertDescription className="text-sm text-amber-800">
                <strong>Quantum Limit:</strong> Maximum 4 cities (16 qubits)
              </AlertDescription>
            </Alert>

            <div className="space-y-2">
              <Label className="text-sm font-semibold">Number of Cities</Label>
              <Input type="number" min="2" max="4" value={numCities} onChange={(e) => setNumCities(Number(e.target.value))} data-testid="num-cities-input" className="border-2" />
            </div>

            <div className="space-y-2">
              <Label className="text-sm font-semibold">Distance Matrix (JSON)</Label>
              <textarea value={distanceMatrix} onChange={(e) => setDistanceMatrix(e.target.value)} placeholder='[[0,10,15],[10,0,20],[15,20,0]]' className="w-full h-24 px-3 py-2 border-2 rounded-md font-mono text-xs focus:ring-2 focus:ring-green-500" data-testid="distance-matrix-input" />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-sm font-semibold">QAOA Layers</Label>
                <Select value={pLayers} onChange={(e) => setPLayers(Number(e.target.value))} className="border-2" data-testid="p-layers-select">
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-sm font-semibold">Iterations</Label>
                <Select value={maxIter} onChange={(e) => setMaxIter(Number(e.target.value))} className="border-2" data-testid="max-iter-select">
                  <option value="30">30</option>
                  <option value="50">50</option>
                  <option value="100">100</option>
                </Select>
              </div>
            </div>

            <div className="flex gap-2">
              <Button onClick={solveProblem} disabled={loading || generating} className="flex-1 bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700 shadow-lg" data-testid="solve-button">
                {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Optimizing...</> : <><Play className="w-4 h-4 mr-2" />Solve TSP</>}
              </Button>
              <Button onClick={generateRandom} disabled={loading || generating} variant="outline" className="border-2 border-green-300" data-testid="generate-button">
                {generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
              </Button>
            </div>

            {error && <Alert variant="destructive"><AlertCircle className="w-4 h-4" /><AlertDescription>{error}</AlertDescription></Alert>}
          </CardContent>
        </Card>

        <div className="lg:col-span-3 space-y-6">
          <Card className="shadow-lg border-2 border-teal-100" data-testid="tsp-results-panel">
            <CardHeader className="bg-gradient-to-r from-teal-600 to-green-600 text-white">
              <CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5" />Optimization Results</CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              {!result && !loading && (
                <div className="text-center py-16 text-gray-400">
                  <MapPin className="w-20 h-20 mx-auto mb-4 opacity-30" />
                  <p className="text-lg">Configure TSP and run QAOA</p>
                </div>
              )}

              {loading && (
                <div className="text-center py-16">
                  <Loader2 className="w-16 h-16 animate-spin mx-auto text-teal-600 mb-4" />
                  <p className="text-lg font-semibold">Running Quantum TSP Optimization...</p>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-gradient-to-br from-teal-500 to-teal-600 text-white p-5 rounded-xl shadow-lg">
                      <p className="text-sm opacity-90">Tour Cost</p>
                      <p className="text-4xl font-bold">{result.tour_cost.toFixed(1)}</p>
                    </div>
                    <div className="bg-gradient-to-br from-green-500 to-green-600 text-white p-5 rounded-xl shadow-lg">
                      <p className="text-sm opacity-90">Approximation</p>
                      <p className="text-4xl font-bold">{(result.classical_comparison.qaoa_approximation_ratio * 100).toFixed(0)}%</p>
                    </div>
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-5 rounded-xl shadow-lg">
                      <p className="text-sm opacity-90">Valid Tour</p>
                      <p className="text-4xl font-bold">{result.is_valid ? '✓' : '✗'}</p>
                    </div>
                  </div>

                  <Card className="border-2 border-teal-200 bg-teal-50">
                    <CardHeader className="pb-3"><CardTitle className="text-base text-teal-900">Optimal Tour</CardTitle></CardHeader>
                    <CardContent>
                      <div className="bg-white p-4 rounded-lg text-center">
                        <p className="text-3xl font-bold text-teal-700">{result.tour_string}</p>
                        <p className="text-sm text-gray-600 mt-2">Distance: {result.tour_cost.toFixed(2)}</p>
                      </div>
                    </CardContent>
                  </Card>

                  {result.convergence_analysis && (
                    <Card className="border-2 border-purple-200">
                      <CardHeader className="bg-purple-50"><CardTitle className="text-base flex items-center gap-2"><BarChart3 className="w-4 h-4" />Convergence</CardTitle></CardHeader>
                      <CardContent className="pt-4">
                        <ResponsiveContainer width="100%" height={200}>
                          <AreaChart data={getConvergenceData()}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="iteration" />
                            <YAxis />
                            <Tooltip />
                            <Area type="monotone" dataKey="cost" stroke="#14b8a6" fill="#5eead4" />
                          </AreaChart>
                        </ResponsiveContainer>
                        <div className="grid grid-cols-3 gap-3 mt-4">
                          <div className="bg-purple-50 p-3 rounded text-center">
                            <p className="text-xs text-gray-600">Initial</p>
                            <p className="text-lg font-bold">{result.convergence_analysis.initial_cost?.toFixed(2)}</p>
                          </div>
                          <div className="bg-purple-50 p-3 rounded text-center">
                            <p className="text-xs text-gray-600">Final</p>
                            <p className="text-lg font-bold">{result.convergence_analysis.final_cost?.toFixed(2)}</p>
                          </div>
                          <div className="bg-green-50 p-3 rounded text-center">
                            <p className="text-xs text-gray-600">Improvement</p>
                            <p className="text-lg font-bold text-green-700">{result.convergence_analysis.improvement?.toFixed(1)}%</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  <Card className="border-2 border-blue-200">
                    <CardHeader className="bg-blue-50"><CardTitle className="text-base">Performance Comparison</CardTitle></CardHeader>
                    <CardContent className="pt-4">
                      <div className="space-y-3">
                        <div className="flex justify-between items-center p-3 bg-teal-50 rounded-lg">
                          <span className="font-semibold">QAOA Result:</span>
                          <Badge className="bg-teal-600">{result.tour_cost.toFixed(2)}</Badge>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                          <span className="font-semibold">Classical (Greedy):</span>
                          <Badge variant="secondary">{result.classical_comparison.classical_cost.toFixed(2)}</Badge>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                          <span className="font-semibold">Optimal:</span>
                          <Badge className="bg-green-600">{result.classical_comparison.optimal_cost.toFixed(2)}</Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </CardContent>
          </Card>

          {result && result.problem_info && (
            <QuantumCircuitVisual 
              numQubits={result.problem_info.num_qubits} 
              pLayers={pLayers}
              problemType="tsp"
            />
          )}
        </div>
      </div>
    </div>
  );
}
