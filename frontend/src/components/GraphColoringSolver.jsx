import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { Badge } from './ui/badge';
import { Loader2, Play, RefreshCw, CheckCircle, AlertCircle, Palette, TrendingUp, BarChart3 } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import QuantumCircuitVisual from './QuantumCircuitVisual';
import { GraphColoringFormula } from './MathFormula';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function GraphColoringSolver() {
  const [numVertices, setNumVertices] = useState(3);
  const [numColors, setNumColors] = useState(2);
  const [edges, setEdges] = useState('0-1,1-2,0-2');
  const [pLayers, setPLayers] = useState(2);
  const [maxIter, setMaxIter] = useState(100);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const generateRandom = async () => {
    setGenerating(true);
    setError(null);
    try {
      const response = await axios.post(`${API}/generate/random`, {
        problem_type: 'graph_coloring',
        size: numVertices,
        additional_params: {
          edge_probability: 0.4
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
      }).filter(e => !isNaN(e[0]) && !isNaN(e[1]));

      if (numVertices > 4) {
        throw new Error('Maximum 4 vertices allowed for quantum simulation');
      }

      const response = await axios.post(`${API}/graph-coloring/solve`, {
        num_vertices: numVertices,
        edges: edgeList,
        num_colors: numColors,
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

  const colorNames = ['Red', 'Blue', 'Green', 'Yellow', 'Purple'];
  const colorClasses = ['bg-red-500', 'bg-blue-500', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500'];

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
      <GraphColoringFormula />

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Input Panel */}
        <Card className="lg:col-span-2 shadow-lg border-2 border-pink-100" data-testid="gc-input-panel">
          <CardHeader className="bg-gradient-to-r from-pink-600 to-purple-600 text-white">
            <CardTitle className="flex items-center gap-2">
              <Palette className="w-5 h-5" />
              Graph Coloring Config
            </CardTitle>
            <CardDescription className="text-pink-100">
              Assign colors to vertices (no adjacent same color)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-6">
            <Alert className="bg-purple-50 border-purple-300">
              <AlertDescription className="text-sm text-purple-800">
                <strong>Limit:</strong> Max 4 vertices, 3 colors
              </AlertDescription>
            </Alert>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="num-vertices">Vertices</Label>
              <Input
                id="num-vertices"
                type="number"
                min="2"
                max="4"
                value={numVertices}
                onChange={(e) => setNumVertices(Number(e.target.value))}
                data-testid="num-vertices-input"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="num-colors">Colors</Label>
              <Input
                id="num-colors"
                type="number"
                min="2"
                max="3"
                value={numColors}
                onChange={(e) => setNumColors(Number(e.target.value))}
                data-testid="num-colors-input"
              />
            </div>
          </div>

            <div className="space-y-2">
              <Label htmlFor="edges" className="text-sm font-semibold">Edges (format: 0-1,1-2,0-2)</Label>
              <Input
                id="edges"
                value={edges}
                onChange={(e) => setEdges(e.target.value)}
                placeholder="0-1,1-2,0-2"
                data-testid="edges-input"
                className="border-2 font-mono text-sm"
              />
              <p className="text-xs text-gray-500">Triangle: 0-1,1-2,0-2</p>
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
                className="flex-1 bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700 shadow-lg"
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
                className="border-2 border-pink-300"
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
        <div className="lg:col-span-3 space-y-6">
          <Card className="shadow-lg border-2 border-purple-100" data-testid="gc-results-panel">
            <CardHeader className="bg-gradient-to-r from-purple-600 to-pink-600 text-white">
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Coloring Results
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              {!result && !loading && (
                <div className="text-center py-16 text-gray-400">
                  <Palette className="w-20 h-20 mx-auto mb-4 opacity-30" />
                  <p className="text-lg">Configure graph and run QAOA</p>
                </div>
              )}

              {loading && (
                <div className="text-center py-16">
                  <Loader2 className="w-16 h-16 animate-spin mx-auto text-purple-600 mb-4" />
                  <p className="text-lg font-semibold">Running Quantum Graph Coloring...</p>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white p-5 rounded-xl shadow-lg">
                      <p className="text-sm opacity-90">Conflicts</p>
                      <p className="text-4xl font-bold">{result.num_conflicts}</p>
                    </div>
                    <div className="bg-gradient-to-br from-pink-500 to-pink-600 text-white p-5 rounded-xl shadow-lg">
                      <p className="text-sm opacity-90">Colors Used</p>
                      <p className="text-4xl font-bold">{result.graph_metrics.num_colors}</p>
                    </div>
                    <div className="bg-gradient-to-br from-green-500 to-green-600 text-white p-5 rounded-xl shadow-lg">
                      <p className="text-sm opacity-90">Valid</p>
                      <p className="text-4xl font-bold">{result.is_valid ? '\u2713' : '\u2717'}</p>
                    </div>
                  </div>

                  <Card className="border-2 border-purple-200 bg-purple-50">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base text-purple-900">Vertex Coloring</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="bg-white p-4 rounded-lg space-y-2">
                        {result.coloring && Object.entries(result.coloring).map(([vertex, color]) => (
                          <div key={vertex} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                            <span className="font-semibold text-lg">Vertex {vertex}</span>
                            <div className="flex items-center gap-2">
                              <div className={`w-8 h-8 rounded-full ${colorClasses[color] || 'bg-gray-400'} shadow-lg`} />
                              <span className="text-sm font-semibold">{colorNames[color] || `Color ${color}`}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {result.convergence_analysis && (
                    <Card className="border-2 border-indigo-200">
                      <CardHeader className="bg-indigo-50">
                        <CardTitle className="text-base flex items-center gap-2">
                          <BarChart3 className="w-4 h-4" />
                          Convergence Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="pt-4">
                        <ResponsiveContainer width="100%" height={200}>
                          <AreaChart data={getConvergenceData()}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="iteration" />
                            <YAxis />
                            <Tooltip />
                            <Area type="monotone" dataKey="cost" stroke="#a855f7" fill="#c084fc" />
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
                    <CardHeader className="bg-blue-50">
                      <CardTitle className="text-base">Performance Comparison</CardTitle>
                    </CardHeader>
                    <CardContent className="pt-4">
                      <div className="space-y-3">
                        <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                          <span className="font-semibold">QAOA Conflicts:</span>
                          <Badge className="bg-purple-600">{result.num_conflicts}</Badge>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                          <span className="font-semibold">Classical Conflicts:</span>
                          <Badge variant="secondary">{result.classical_comparison.classical_conflicts}</Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </CardContent>
          </Card>

          {result && result.graph_metrics && (
            <QuantumCircuitVisual 
              numQubits={result.graph_metrics.num_qubits} 
              pLayers={pLayers}
              problemType="graph_coloring"
            />
          )}
        </div>
      </div>
    </div>
  );
}
