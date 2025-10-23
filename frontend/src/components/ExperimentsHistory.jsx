import { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Trash2, RefreshCw } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function ExperimentsHistory({ onRefresh }) {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    try {
      const response = await axios.get(`${API}/experiments?limit=50`);
      setExperiments(response.data);
      setLoading(false);
      if (onRefresh) onRefresh();
    } catch (error) {
      console.error('Error fetching experiments:', error);
      setLoading(false);
    }
  };

  const deleteExperiment = async (id) => {
    try {
      await axios.delete(`${API}/experiments/${id}`);
      fetchExperiments();
    } catch (error) {
      console.error('Error deleting experiment:', error);
    }
  };

  const getProblemBadge = (type) => {
    const colors = {
      maxcut: 'default',
      tsp: 'secondary',
      vrp: 'success',
      graph_coloring: 'destructive'
    };
    return <Badge variant={colors[type] || 'default'}>{type.toUpperCase()}</Badge>;
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>Experiment History</CardTitle>
            <CardDescription>View and manage past QAOA experiments</CardDescription>
          </div>
          <Button onClick={fetchExperiments} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="text-center py-8 text-gray-400">Loading experiments...</div>
        ) : experiments.length === 0 ? (
          <div className="text-center py-8 text-gray-400">No experiments yet. Run a solver to create one!</div>
        ) : (
          <div className="space-y-3">
            {experiments.map((exp) => (
              <div
                key={exp.id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    {getProblemBadge(exp.problem_type)}
                    <span className="text-sm text-gray-500">
                      {new Date(exp.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600">
                    p={exp.parameters?.p_layers || 'N/A'} layers, 
                    {exp.parameters?.max_iter || 'N/A'} iterations
                  </div>
                </div>
                <Button
                  onClick={() => deleteExperiment(exp.id)}
                  variant="ghost"
                  size="sm"
                  className="text-red-600 hover:text-red-700"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
