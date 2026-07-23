using Microsoft.UI.Xaml;

namespace LightVideoEnhancer_WinUI;

public partial class App : Application
{
    public App()
    {
        InitializeComponent();
    }

    public MainWindow? MainWindow { get; private set; }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        MainWindow = new MainWindow();
        MainWindow.Activate();
    }
}
