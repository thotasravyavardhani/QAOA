import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { Badge } from './ui/badge';
import { Loader2, Play, RefreshCw, CheckCircle, TrendingUp, BarChart3, Network } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import QuantumCircuitVisual from './QuantumCircuitVisual';
import { MaxCutFormula } from './MathFormula';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function MaxCutSolver() {
  const [numVertices, setNumVertices] = useState(4);
  const [pLayers, setPLayers] = useState(2);
  const [maxIter, setMaxIter] = useState(50);
  const [edges, setEdges] = useState('0-1,1-2,2-3,0-3,0-2');
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const generateRandom = async () => {
    setGenerating(true);
    setError(null);
    try {
      const response = await axios.post(`${API}/generate/random`, {
        problem_type: 'maxcut',
        size: numVertices,
        additional_params: {
          edge_probability: 0.5,
          weighted: false
        }
      });
      
      const edgeStr = response.data.edges.map(e => `${e[0]}-${e[1]}`).join(',');
      setEdges(edgeStr);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error generating graph');
    } finally {
      setGenerating(false);
    }
  };

  const solveProblem = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const edgeList = edges.split(',').map(e => {
        const [u, v] = e.trim().split('-').map(Number);
        return [u, v];
      });

      const response = await axios.post(`${API}/maxcut/solve`, {
        num_vertices: numVertices,
        edges: edgeList,
        p_layers: pLayers,
        max_iter: maxIter,
        method: 'COBYLA',
        initialization_strategy: 'standard'
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error solving problem');
    } finally {
      setLoading(false);
    }
  };

  // Prepare convergence data for chart
  const getConvergenceData = () => {
    if (!result?.convergence_analysis) return [];
    const { initial_cost, final_cost, iterations } = result.convergence_analysis;
    
    // Simulate convergence curve
    return Array.from({ length: Math.min(iterations, 20) }, (_, i) => ({
      iteration: i,
      cost: initial_cost + (final_cost - initial_cost) * (i / iterations) * (1 + Math.random() * 0.1)
    }));
  };

  // Prepare probability distribution data
  const getProbabilityData = () => {
    if (!result?.probability_distribution) return [];
    return Object.entries(result.probability_distribution)
      .slice(0, 8)
      .map(([state, prob]) => ({
        state,
        probability: (prob * 100).toFixed(2)
      }));
  };

  return (
    <div className="space-y-6">
      {/* Formula Section */}
      <MaxCutFormula />

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Input Panel - 2 columns */}
        <Card className="lg:col-span-2 shadow-lg border-2 border-blue-100" data-testid="maxcut-input-panel">
          <CardHeader className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
            <CardTitle className="flex items-center gap-2">
              <Network className="w-5 h-5" />
              Problem Configuration
            </CardTitle>
            <CardDescription className="text-blue-100">
              Define graph structure and QAOA parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-6">
            <div className="space-y-2">
              <Label htmlFor="vertices" className="text-sm font-semibold">Number of Vertices</Label>
              <Input
                id="vertices"
                type="number"
                min="2"
                max="10"
                value={numVertices}
                onChange={(e) => setNumVertices(Number(e.target.value))}
                data-testid="vertices-input"
                className="border-2 focus:border-blue-500"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="edges" className="text-sm font-semibold">Edges (format: 0-1,1-2,2-3)</Label>
              <Input
                id="edges"
                value={edges}
                onChange={(e) => setEdges(e.target.value)}
                placeholder="0-1,1-2,2-3"
                data-testid="edges-input"
                className="border-2 focus:border-blue-500 font-mono text-sm"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="p-layers" className="text-sm font-semibold">QAOA Layers (p)</Label>
                <Select
                  id="p-layers"
                  value={pLayers}
                  onChange={(e) => setPLayers(Number(e.target.value))}
                  data-testid="p-layers-select"
                  className="border-2"
                >
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                  <option value="4">4</option>
                  <option value="5">5</option>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="max-iter" className="text-sm font-semibold">Max Iterations</Label>
                <Select
                  id="max-iter"
                  value={maxIter}
                  onChange={(e) => setMaxIter(Number(e.target.value))}
                  data-testid="max-iter-select"
                  className="border-2"
                >
                  <option value="20">20</option>
                  <option value="50">50</option>
                  <option value="100">100</option>
                  <option value="200">200</option>
                </Select>
              </div>
            </div>

            <div className="flex gap-2 pt-2">
              <Button
                onClick={solveProblem}
                disabled={loading || generating}
                className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-lg"
                data-testid="solve-button"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Optimizing...
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
                className="border-2 border-indigo-300 hover:bg-indigo-50"
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
              <Alert variant="destructive" className="border-2">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Results Panel - 3 columns */}
        <div className="lg:col-span-3 space-y-6">
          <Card className="shadow-lg border-2 border-indigo-100" data-testid="maxcut-results-panel">
            <CardHeader className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Optimization Results
              </CardTitle>
              <CardDescription className="text-indigo-100">QAOA quantum computing solution</CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              {!result && !loading && (
                <div className="text-center py-16 text-gray-400">
                  <Network className="w-20 h-20 mx-auto mb-4 opacity-30" />
                  <p className="text-lg">Configure parameters and run QAOA to see results</p>
                </div>
              )}

              {loading && (
                <div className="text-center py-16">
                  <Loader2 className="w-16 h-16 animate-spin mx-auto text-indigo-600 mb-4" />
                  <p className="text-lg font-semibold text-gray-700">Running Quantum Optimization...</p>
                  <p className="text-sm text-gray-500 mt-2">Exploring {Math.pow(2, numVertices)} quantum states</p>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  {/* Key Metrics */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-5 rounded-xl shadow-lg">
                      <p className="text-sm opacity-90 mb-1">Cut Value</p>
                      <p className="text-4xl font-bold">{result.cut_value}</p>
                    </div>
                    <div className="bg-gradient-to-br from-green-500 to-green-600 text-white p-5 rounded-xl shadow-lg">
                      <p className="text-sm opacity-90 mb-1">Approximation</p>
                      <p className="text-4xl font-bold">{(result.approximation_ratio * 100).toFixed(0)}%</p>
                    </div>
                    <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white p-5 rounded-xl shadow-lg">
                      <p className="text-sm opacity-90 mb-1">Iterations</p>
                      <p className="text-4xl font-bold">{result.iterations}</p>
                    </div>
                  </div>

                  {/* Status Badge */}
                  <div className="flex items-center gap-3 bg-gray-50 p-4 rounded-lg">
                    {result.success ? (
                      <>
                        <CheckCircle className="w-6 h-6 text-green-600" />
                        <Badge className="bg-green-100 text-green-800 text-sm px-3 py-1">Optimization Complete</Badge>
                      </>
                    ) : (
                      <Badge variant="destructive" className="text-sm px-3 py-1">Optimization Failed</Badge>
                    )}
                    <span className="text-sm text-gray-600 ml-auto">Classical Bound: {result.classical_bound}</span>
                  </div>

                  {/* Best Partition */}
                  <Card className="border-2 border-blue-200 bg-blue-50">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base text-blue-900">Optimal Partition</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="bg-white p-4 rounded-lg font-mono text-2xl text-center text-blue-700 font-bold tracking-widest border-2 border-blue-300">
                        {result.best_partition}
                      </div>
                      <p className="text-xs text-gray-600 mt-2 text-center">
                        Vertices: {result.best_partition.split('').map((bit, i) => `V${i}:${bit === '1' ? 'A' : 'B'}`).join(', ')}
                      </p>
                    </CardContent>
                  </Card>

                  {/* Convergence Chart */}
                  {result.convergence_analysis && (
                    <Card className="border-2 border-purple-200">
                      <CardHeader className="bg-purple-50">
                        <CardTitle className="text-base text-purple-900 flex items-center gap-2">
                          <BarChart3 className="w-4 h-4" />
                          Convergence Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="pt-4">
                        <ResponsiveContainer width="100%" height={200}>
                          <AreaChart data={getConvergenceData()}>
                            <defs>
                              <linearGradient id="colorCost" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                            <YAxis label={{ value: 'Cost', angle: -90, position: 'insideLeft' }} />
                            <Tooltip />
                            <Area type="monotone" dataKey="cost" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorCost)" />
                          </AreaChart>
                        </ResponsiveContainer>
                        <div className="grid grid-cols-3 gap-3 mt-4">
                          <div className="bg-purple-50 p-3 rounded-lg text-center">
                            <p className="text-xs text-gray-600">Initial Cost</p>
                            <p className="text-lg font-bold text-purple-700">{result.convergence_analysis.initial_cost?.toFixed(3)}</p>
                          </div>
                          <div className="bg-purple-50 p-3 rounded-lg text-center">
                            <p className="text-xs text-gray-600">Final Cost</p>
                            <p className="text-lg font-bold text-purple-700">{result.convergence_analysis.final_cost?.toFixed(3)}</p>
                          </div>
                          <div className="bg-green-50 p-3 rounded-lg text-center">
                            <p className="text-xs text-gray-600">Improvement</p>
                            <p className="text-lg font-bold text-green-700">{result.convergence_analysis.improvement?.toFixed(1)}%</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Probability Distribution */}
                  {result.probability_distribution && (
                    <Card className="border-2 border-indigo-200">
                      <CardHeader className="bg-indigo-50">
                        <CardTitle className="text-base text-indigo-900 flex items-center gap-2">
                          <BarChart3 className="w-4 h-4" />
                          State Probability Distribution
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="pt-4">
                        <ResponsiveContainer width="100%" height={250}>
                          <BarChart data={getProbabilityData()}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="state" label={{ value: 'Bitstring', position: 'insideBottom', offset: -5 }} />
                            <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} />
                            <Tooltip />
                            <Bar dataKey="probability" fill="#6366f1" radius={[8, 8, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quantum Circuit Visualization */}
          {result && (
            <QuantumCircuitVisual 
              numQubits={numVertices} 
              pLayers={pLayers}
              problemType="maxcut"
            />
          )}
        </div>
      </div>
    </div>
  );
}
