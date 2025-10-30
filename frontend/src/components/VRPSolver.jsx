import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { Badge } from './ui/badge';
import { Loader2, Play, RefreshCw, CheckCircle, AlertCircle, Truck } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function VRPSolver() {
  const [numCustomers, setNumCustomers] = useState(2);
  const [numVehicles, setNumVehicles] = useState(2);
  const [capacity, setCapacity] = useState(100);
  const [pLayers, setPLayers] = useState(2);
  const [maxIter, setMaxIter] = useState(50);
  const [distanceMatrix, setDistanceMatrix] = useState('[[0,10,15],[10,0,20],[15,20,0]]');
  const [demands, setDemands] = useState('[30,40]');
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const generateRandom = async () => {
    setGenerating(true);
    setError(null);
    try {
      const response = await axios.post(`${API}/generate/random`, {
        problem_type: 'vrp',
        size: numCustomers,
        additional_params: {
          num_vehicles: numVehicles,
          capacity: capacity
        }
      });
      
      setDistanceMatrix(JSON.stringify(response.data.distance_matrix));
      setDemands(JSON.stringify(response.data.demands));
    } catch (err) {
      setError(err.response?.data?.detail || 'Error generating VRP instance');
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
      const demandsList = JSON.parse(demands);
      
      if (demandsList.length > 3) {
        throw new Error('Maximum 3 customers allowed for quantum simulation');
      }

      const response = await axios.post(`${API}/vrp/solve`, {
        distance_matrix: matrix,
        demands: demandsList,
        vehicle_capacity: capacity,
        num_vehicles: numVehicles,
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
      <Card data-testid="vrp-input-panel">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Truck className="w-5 h-5" />
            Vehicle Routing Problem (VRP)
          </CardTitle>
          <CardDescription>
            Optimize routes for multiple vehicles serving customers
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <AlertDescription className="text-sm">
              <strong>Limit:</strong> Maximum 3 customers, 2 vehicles for quantum simulation
            </AlertDescription>
          </Alert>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="num-customers">Customers</Label>
              <Input
                id="num-customers"
                type="number"
                min="1"
                max="3"
                value={numCustomers}
                onChange={(e) => setNumCustomers(Number(e.target.value))}
                data-testid="num-customers-input"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="num-vehicles">Vehicles</Label>
              <Input
                id="num-vehicles"
                type="number"
                min="1"
                max="2"
                value={numVehicles}
                onChange={(e) => setNumVehicles(Number(e.target.value))}
                data-testid="num-vehicles-input"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="capacity">Vehicle Capacity</Label>
            <Input
              id="capacity"
              type="number"
              min="10"
              value={capacity}
              onChange={(e) => setCapacity(Number(e.target.value))}
              data-testid="capacity-input"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="distance-matrix">Distance Matrix (depot + customers)</Label>
            <textarea
              id="distance-matrix"
              value={distanceMatrix}
              onChange={(e) => setDistanceMatrix(e.target.value)}
              placeholder='[[0,10,15],[10,0,20],[15,20,0]]'
              className="w-full h-24 px-3 py-2 border rounded-md font-mono text-sm"
              data-testid="distance-matrix-input"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="demands">Customer Demands (JSON)</Label>
            <Input
              id="demands"
              value={demands}
              onChange={(e) => setDemands(e.target.value)}
              placeholder='[30,40]'
              className="font-mono"
              data-testid="demands-input"
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
      <Card data-testid="vrp-results-panel">
        <CardHeader>
          <CardTitle>Results</CardTitle>
          <CardDescription>Vehicle routing optimization results</CardDescription>
        </CardHeader>
        <CardContent>
          {!result && !loading && (
            <div className="text-center py-12 text-gray-400">
              <Truck className="w-16 h-16 mx-auto mb-4" />
              <p>Configure VRP and run QAOA to see results</p>
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
                  {result.is_feasible && result.is_valid ? (
                    <>
                      <CheckCircle className="w-5 h-5 text-green-600" />
                      <Badge className="bg-green-100 text-green-800">Valid Solution</Badge>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-5 h-5 text-orange-600" />
                      <Badge className="bg-orange-100 text-orange-800">
                        {!result.is_feasible ? 'Capacity Exceeded' : 'Approximate'}
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
                  <p className="text-sm text-gray-600">Total Cost</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {result.total_cost.toFixed(2)}
                  </p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-gray-600">Improvement</p>
                  <p className="text-3xl font-bold text-green-600">
                    {result.classical_comparison.improvement.toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Routes */}
              <div>
                <Label className="mb-2 block">Vehicle Routes</Label>
                <div className="bg-gray-50 p-4 rounded space-y-2">
                  <p className="font-mono text-sm text-blue-700">
                    {result.routes_string}
                  </p>
                </div>
              </div>

              {/* Vehicle Loads */}
              {result.vehicle_loads && result.vehicle_loads.length > 0 && (
                <div>
                  <Label className="mb-2 block">Vehicle Loads</Label>
                  <div className="space-y-2">
                    {result.vehicle_loads.map((load, idx) => (
                      <div key={idx} className="flex justify-between items-center bg-gray-50 p-2 rounded">
                        <span className="text-sm">Vehicle {idx}:</span>
                        <Badge variant={load <= capacity ? 'default' : 'destructive'}>
                          {load.toFixed(1)} / {capacity}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Performance Comparison */}
              <div>
                <Label className="mb-2 block">Performance Comparison</Label>
                <div className="space-y-2 bg-gray-50 p-4 rounded">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">QAOA Result:</span>
                    <Badge variant="default">{result.total_cost.toFixed(2)}</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Classical (Greedy):</span>
                    <Badge variant="secondary">
                      {result.classical_comparison.classical_cost.toFixed(2)}
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

              {/* Problem Info */}
              <div className="text-xs text-gray-500 space-y-1">
                <div className="flex justify-between">
                  <span>Customers:</span>
                  <span>{result.problem_info.num_customers}</span>
                </div>
                <div className="flex justify-between">
                  <span>Vehicles:</span>
                  <span>{result.problem_info.num_vehicles}</span>
                </div>
                <div className="flex justify-between">
                  <span>Qubits Used:</span>
                  <span>{result.problem_info.num_qubits}</span>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
