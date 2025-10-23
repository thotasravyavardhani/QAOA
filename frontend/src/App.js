import { useState, useEffect } from 'react';
import '@/App.css';
import axios from 'axios';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import MaxCutSolver from './components/MaxCutSolver';
import TSPSolver from './components/TSPSolver';
import VRPSolver from './components/VRPSolver';
import GraphColoringSolver from './components/GraphColoringSolver';
import ExperimentsHistory from './components/ExperimentsHistory';
import Documentation from './components/Documentation';
import { Activity, Sparkles, FileText, History } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('maxcut');

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/stats`);
      setStats(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-xl">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
                <Sparkles className="w-10 h-10" />
                QAOA Optimization Platform
              </h1>
              <p className="text-blue-100 text-lg">
                Quantum Approximate Optimization Algorithm for NP-Hard Problems
              </p>
            </div>
            <div className="flex gap-4">
              <Card className="bg-white/10 border-white/20 text-white">
                <CardHeader className="pb-2">
                  <CardDescription className="text-blue-100">Total Experiments</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-bold">{stats?.total_experiments || 0}</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-6 lg:w-auto lg:inline-flex bg-white shadow-md">
            <TabsTrigger value="maxcut" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Max-Cut
            </TabsTrigger>
            <TabsTrigger value="tsp" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              TSP
            </TabsTrigger>
            <TabsTrigger value="vrp" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              VRP
            </TabsTrigger>
            <TabsTrigger value="coloring" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Graph Coloring
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <History className="w-4 h-4" />
              History
            </TabsTrigger>
            <TabsTrigger value="docs" className="flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Docs
            </TabsTrigger>
          </TabsList>

          <TabsContent value="maxcut" className="space-y-4">
            <MaxCutSolver />
          </TabsContent>

          <TabsContent value="tsp" className="space-y-4">
            <TSPSolver />
          </TabsContent>

          <TabsContent value="vrp" className="space-y-4">
            <VRPSolver />
          </TabsContent>

          <TabsContent value="coloring" className="space-y-4">
            <GraphColoringSolver />
          </TabsContent>

          <TabsContent value="history" className="space-y-4">
            <ExperimentsHistory onRefresh={fetchStats} />
          </TabsContent>

          <TabsContent value="docs" className="space-y-4">
            <Documentation />
          </TabsContent>
        </Tabs>
      </div>

      {/* Footer */}
      <div className="bg-white border-t mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-sm text-gray-600">
              <p className="font-semibold">QAOA Research Platform</p>
              <p className="text-xs">Solving NP-hard problems with quantum algorithms</p>
            </div>
            <div className="flex gap-4 flex-wrap">
              <Badge variant="outline">Qiskit 1.3.1</Badge>
              <Badge variant="outline">FastAPI</Badge>
              <Badge variant="outline">React 19</Badge>
              <Badge variant="outline">MongoDB</Badge>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
