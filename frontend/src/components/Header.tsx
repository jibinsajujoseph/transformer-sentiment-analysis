import './Header.css';

const Header = () => {
  return (
    <header className="app-header">
      <div className="logo-container">
        <div className="logo-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2v20"></path>
            <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
          </svg>
        </div>
        <h1 className="app-title">Sentiment <span className="text-gradient">Model Arena</span></h1>
      </div>
      <p className="app-description">
        Compare a custom PyTorch Transformer against a fine-tuned DistilRoBERTa model side-by-side.
      </p>
    </header>
  );
};

export default Header;
