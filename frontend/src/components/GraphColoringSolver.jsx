import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { Badge } from './ui/badge';
import { Loader2, Play, RefreshCw, CheckCircle, AlertCircle, Palette } from 'lucide-react';

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
  const colorClasses = [
    'bg-red-500',
    'bg-blue-500',
    'bg-green-500',
    'bg-yellow-500',
    'bg-purple-500'
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Panel */}
      <Card data-testid="gc-input-panel">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Palette className="w-5 h-5" />
            Graph Coloring Problem
          </CardTitle>
          <CardDescription>
            Assign colors such that no adjacent vertices share the same color
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <AlertDescription className="text-sm">
              <strong>Limit:</strong> Maximum 4 vertices, 3 colors for quantum simulation
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
            <Label htmlFor="edges">Edges (format: 0-1,1-2,0-2)</Label>
            <Input
              id="edges"
              value={edges}
              onChange={(e) => setEdges(e.target.value)}
              placeholder="0-1,1-2,0-2"
              data-testid="edges-input"
            />
            <p className="text-xs text-gray-500">
              Example: 0-1,1-2,0-2 (creates a triangle)
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
                <option value="50">50</option>
                <option value="100">100</option>
                <option value="200">200</option>
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
      <Card data-testid="gc-results-panel">
        <CardHeader>
          <CardTitle>Results</CardTitle>
          <CardDescription>Graph coloring optimization results</CardDescription>
        </CardHeader>
        <CardContent>
          {!result && !loading && (
            <div className="text-center py-12 text-gray-400">
              <Palette className="w-16 h-16 mx-auto mb-4" />
              <p>Configure graph and run QAOA to see results</p>
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
              {/* Status */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {result.is_valid ? (
                    <>
                      <CheckCircle className="w-5 h-5 text-green-600" />
                      <Badge className="bg-green-100 text-green-800">Valid Coloring</Badge>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-5 h-5 text-orange-600" />
                      <Badge className="bg-orange-100 text-orange-800">
                        {result.num_conflicts} Conflict{result.num_conflicts !== 1 ? 's' : ''}
                      </Badge>
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
                  <p className="text-sm text-gray-600">Conflicts</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {result.num_conflicts}
                  </p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Colors Used</p>
                  <p className="text-3xl font-bold text-green-600">
                    {result.graph_metrics.num_colors}
                  </p>
                </div>
              </div>

              {/* Vertex Coloring */}
              <div>
                <Label className="mb-2 block">Vertex Coloring</Label>
                <div className="bg-gray-50 p-4 rounded space-y-2">
                  {result.coloring && Object.entries(result.coloring).map(([vertex, color]) => (
                    <div key={vertex} className="flex items-center justify-between p-2 bg-white rounded">
                      <span className="font-semibold">Vertex {vertex}:</span>
                      <div className="flex items-center gap-2">
                        <div className={`w-4 h-4 rounded-full ${colorClasses[color] || 'bg-gray-400'}`} />
                        <span className="text-sm">{colorNames[color] || `Color ${color}`}</span>
                      </div>
                    </div>
                  ))}
                  <p className="text-sm text-gray-600 text-center mt-3 pt-3 border-t">
                    {result.coloring_string}
                  </p>
                </div>
              </div>

              {/* Performance Comparison */}
              <div>
                <Label className="mb-2 block">Performance Comparison</Label>
                <div className="space-y-2 bg-gray-50 p-4 rounded">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">QAOA Conflicts:</span>
                    <Badge variant="default">{result.num_conflicts}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Classical (Greedy) Conflicts:</span>
                    <Badge variant="secondary">
                      {result.classical_comparison.classical_conflicts}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Improvement:</span>
                    <Badge className="bg-green-100 text-green-800">
                      {result.classical_comparison.improvement >= 0 ? '+' : ''}
                      {result.classical_comparison.improvement}
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Convergence */}
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

              {/* Graph Info */}
              <div className="text-xs text-gray-500 space-y-1">
                <div className="flex justify-between">
                  <span>Vertices:</span>
                  <span>{result.graph_metrics.num_vertices}</span>
                </div>
                <div className="flex justify-between">
                  <span>Edges:</span>
                  <span>{result.graph_metrics.num_edges}</span>
                </div>
                <div className="flex justify-between">
                  <span>Qubits Used:</span>
                  <span>{result.graph_metrics.num_qubits}</span>
                </div>
                <div className="flex justify-between">
                  <span>Chromatic Bound:</span>
                  <span>{result.graph_metrics.chromatic_bound}</span>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
