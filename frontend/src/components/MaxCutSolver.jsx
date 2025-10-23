import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { Badge } from './ui/badge';
import { Loader2, Play, RefreshCw, CheckCircle } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

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
        method: 'COBYLA'
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error solving problem');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Panel */}
      <Card data-testid="maxcut-input-panel">
        <CardHeader>
          <CardTitle>Max-Cut Problem Configuration</CardTitle>
          <CardDescription>
            Partition graph vertices to maximize edges between sets
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="vertices">Number of Vertices</Label>
            <Input
              id="vertices"
              type="number"
              min="2"
              max="10"
              value={numVertices}
              onChange={(e) => setNumVertices(Number(e.target.value))}
              data-testid="vertices-input"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="edges">Edges (format: 0-1,1-2,2-3)</Label>
            <Input
              id="edges"
              value={edges}
              onChange={(e) => setEdges(e.target.value)}
              placeholder="0-1,1-2,2-3"
              data-testid="edges-input"
            />
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
                <option value="4">4</option>
                <option value="5">5</option>
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
                <option value="20">20</option>
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
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Results Panel */}
      <Card data-testid="maxcut-results-panel">
        <CardHeader>
          <CardTitle>Results</CardTitle>
          <CardDescription>QAOA optimization results and analysis</CardDescription>
        </CardHeader>
        <CardContent>
          {!result && !loading && (
            <div className="text-center py-12 text-gray-400">
              <p>Configure and run QAOA to see results</p>
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
              {/* Summary */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Cut Value</p>
                  <p className="text-3xl font-bold text-blue-600">{result.cut_value}</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Approximation Ratio</p>
                  <p className="text-3xl font-bold text-green-600">
                    {(result.approximation_ratio * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Status */}
              <div className="flex items-center gap-2">
                {result.success ? (
                  <>
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <Badge variant="success">Success</Badge>
                  </>
                ) : (
                  <Badge variant="destructive">Failed</Badge>
                )}
                <span className="text-sm text-gray-600">
                  {result.iterations} iterations
                </span>
              </div>

              {/* Partition */}
              <div>
                <Label className="mb-2 block">Best Partition</Label>
                <div className="bg-gray-50 p-3 rounded font-mono text-sm">
                  {result.best_partition}
                </div>
              </div>

              {/* Comparison */}
              <div>
                <Label className="mb-2 block">Performance Comparison</Label>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">QAOA Result:</span>
                    <Badge>{result.cut_value}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Classical Bound:</span>
                    <Badge variant="secondary">{result.classical_bound}</Badge>
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
                      <span className="font-mono">{result.convergence_analysis.initial_cost?.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Final Cost:</span>
                      <span className="font-mono">{result.convergence_analysis.final_cost?.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Improvement:</span>
                      <Badge variant="success">{result.convergence_analysis.improvement?.toFixed(2)}%</Badge>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
