import { useParams, useNavigate } from 'react-router-dom';
import ProjectEditorPage from './ProjectEditorPage';

/**
 * Simple wrapper that extracts projectId from URL and handles navigation
 */
export default function ProjectEditorWrapper() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();

  const handleGoBack = () => {
    navigate('/projects');
  };

  if (!projectId) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-800">Invalid Project</h2>
          <p className="text-gray-600 mt-2">No project ID was provided.</p>
          <button
            onClick={handleGoBack}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            ← Back to Projects
          </button>
        </div>
      </div>
    );
  }

  return <ProjectEditorPage projectId={projectId} onGoBack={handleGoBack} />;
}