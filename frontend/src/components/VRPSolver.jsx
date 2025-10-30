import { useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { Badge } from './ui/badge';
import { Loader2, Play, RefreshCw, Truck, TrendingUp, BarChart3 } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import QuantumCircuitVisual from './QuantumCircuitVisual';
import { VRPFormula } from './MathFormula';

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
        additional_params: { num_vehicles: numVehicles, capacity: capacity }
      });
      setDistanceMatrix(JSON.stringify(response.data.distance_matrix));
      setDemands(JSON.stringify(response.data.demands));
    } catch (err) {
      setError(err.response?.data?.detail || 'Error generating VRP');
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
      if (demandsList.length > 3) throw new Error('Maximum 3 customers allowed');

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
      setError(err.response?.data?.detail || err.message || 'Error');
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
      <VRPFormula />
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <Card className="lg:col-span-2 shadow-lg border-2 border-orange-100" data-testid="vrp-input-panel">
          <CardHeader className="bg-gradient-to-r from-orange-600 to-amber-600 text-white">
            <CardTitle className="flex items-center gap-2"><Truck className="w-5 h-5" />VRP Configuration</CardTitle>
            <CardDescription className="text-orange-100">Optimize vehicle routes</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pt-6">
            <Alert className="bg-amber-50 border-amber-300"><AlertDescription className="text-sm text-amber-800"><strong>Limit:</strong> Max 3 customers, 2 vehicles</AlertDescription></Alert>
            <div className="grid grid-cols-2 gap-4">
              <div><Label className="text-sm font-semibold">Customers</Label><Input type="number" min="1" max="3" value={numCustomers} onChange={(e) => setNumCustomers(Number(e.target.value))} className="border-2" data-testid="num-customers-input" /></div>
              <div><Label className="text-sm font-semibold">Vehicles</Label><Input type="number" min="1" max="2" value={numVehicles} onChange={(e) => setNumVehicles(Number(e.target.value))} className="border-2" data-testid="num-vehicles-input" /></div>
            </div>
            <div><Label className="text-sm font-semibold">Vehicle Capacity</Label><Input type="number" min="10" value={capacity} onChange={(e) => setCapacity(Number(e.target.value))} className="border-2" data-testid="capacity-input" /></div>
            <div><Label className="text-sm font-semibold">Distance Matrix</Label><textarea value={distanceMatrix} onChange={(e) => setDistanceMatrix(e.target.value)} className="w-full h-20 px-3 py-2 border-2 rounded-md font-mono text-xs" data-testid="distance-matrix-input" /></div>
            <div><Label className="text-sm font-semibold">Demands</Label><Input value={demands} onChange={(e) => setDemands(e.target.value)} className="font-mono border-2" data-testid="demands-input" /></div>
            <div className="grid grid-cols-2 gap-4">
              <div><Label className="text-sm font-semibold">Layers</Label><Select value={pLayers} onChange={(e) => setPLayers(Number(e.target.value))} className="border-2" data-testid="p-layers-select"><option value="1">1</option><option value="2">2</option><option value="3">3</option></Select></div>
              <div><Label className="text-sm font-semibold">Iterations</Label><Select value={maxIter} onChange={(e) => setMaxIter(Number(e.target.value))} className="border-2" data-testid="max-iter-select"><option value="30">30</option><option value="50">50</option><option value="100">100</option></Select></div>
            </div>
            <div className="flex gap-2">
              <Button onClick={solveProblem} disabled={loading || generating} className="flex-1 bg-gradient-to-r from-orange-600 to-amber-600 shadow-lg" data-testid="solve-button">{loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Optimizing...</> : <><Play className="w-4 h-4 mr-2" />Solve VRP</>}</Button>
              <Button onClick={generateRandom} disabled={loading || generating} variant="outline" className="border-2 border-orange-300" data-testid="generate-button">{generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}</Button>
            </div>
            {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
          </CardContent>
        </Card>

        <div className="lg:col-span-3 space-y-6">
          <Card className="shadow-lg border-2 border-amber-100" data-testid="vrp-results-panel">
            <CardHeader className="bg-gradient-to-r from-amber-600 to-orange-600 text-white"><CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5" />Results</CardTitle></CardHeader>
            <CardContent className="pt-6">
              {!result && !loading && <div className="text-center py-16 text-gray-400"><Truck className="w-20 h-20 mx-auto mb-4 opacity-30" /><p className="text-lg">Configure VRP and run QAOA</p></div>}
              {loading && <div className="text-center py-16"><Loader2 className="w-16 h-16 animate-spin mx-auto text-orange-600 mb-4" /><p className="text-lg font-semibold">Running Quantum VRP Optimization...</p></div>}
              {result && (
                <div className="space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-gradient-to-br from-orange-500 to-orange-600 text-white p-5 rounded-xl shadow-lg"><p className="text-sm opacity-90">Total Cost</p><p className="text-4xl font-bold">{result.total_cost.toFixed(1)}</p></div>
                    <div className="bg-gradient-to-br from-green-500 to-green-600 text-white p-5 rounded-xl shadow-lg"><p className="text-sm opacity-90">Improvement</p><p className="text-4xl font-bold">{result.classical_comparison.improvement.toFixed(0)}%</p></div>
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-5 rounded-xl shadow-lg"><p className="text-sm opacity-90">Feasible</p><p className="text-4xl font-bold">{result.is_feasible ? '✓' : '✗'}</p></div>
                  </div>
                  <Card className="border-2 border-orange-200 bg-orange-50"><CardHeader className="pb-3"><CardTitle className="text-base text-orange-900">Vehicle Routes</CardTitle></CardHeader><CardContent><div className="bg-white p-4 rounded-lg text-center"><p className="text-2xl font-bold text-orange-700">{result.routes_string}</p></div></CardContent></Card>
                  {result.vehicle_loads && result.vehicle_loads.length > 0 && (
                    <Card className="border-2 border-blue-200"><CardHeader className="pb-3"><CardTitle className="text-base">Vehicle Loads</CardTitle></CardHeader><CardContent><div className="space-y-2">{result.vehicle_loads.map((load, idx) => (<div key={idx} className="flex justify-between items-center bg-gray-50 p-3 rounded-lg"><span className="font-semibold">Vehicle {idx}:</span><Badge variant={load <= capacity ? 'default' : 'destructive'}>{load.toFixed(1)} / {capacity}</Badge></div>))}</div></CardContent></Card>
                  )}
                  {result.convergence_analysis && (
                    <Card className="border-2 border-purple-200"><CardHeader className="bg-purple-50"><CardTitle className="text-base flex items-center gap-2"><BarChart3 className="w-4 h-4" />Convergence</CardTitle></CardHeader><CardContent className="pt-4"><ResponsiveContainer width="100%" height={180}><AreaChart data={getConvergenceData()}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="iteration" /><YAxis /><Tooltip /><Area type="monotone" dataKey="cost" stroke="#f97316" fill="#fb923c" /></AreaChart></ResponsiveContainer></CardContent></Card>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
          {result && result.problem_info && <QuantumCircuitVisual numQubits={result.problem_info.num_qubits} pLayers={pLayers} problemType="vrp" />}
        </div>
      </div>
    </div>
  );
}
